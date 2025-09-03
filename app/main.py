# -*- coding: utf-8 -*-
"""
Управляющий бот + сканер каналов на Telethon.
- bot_client: Telegram bot по BOT_TOKEN (команды /start /on /off /add_channel /list_channels /remove_channel /set_reply /status /whoami)
- user_client: обычный аккаунт (api_id/api_hash) слушает каналы и шлёт ЛС контактам
- Ключевые слова + fallback на Gemini (google-generativeai)
"""

import os, re, json, random, asyncio, logging, base64, pathlib
from typing import Dict, Any, List, Set

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors.rpcerrorlist import (
    FloodWaitError, PeerIdInvalidError, UsernameInvalidError,
    ChatWriteForbiddenError, UserIsBlockedError, UserPrivacyRestrictedError
)

# -------- Gemini --------
import google.generativeai as genai

# -------- Конфиг/ENV --------
load_dotenv()

BOT_TOKEN  = os.getenv("BOT_TOKEN", "").strip()
API_ID     = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH   = os.getenv("TELEGRAM_API_HASH", "").strip()

STATE_FILE   = os.getenv("STATE_FILE", "./data/state.json")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "./.sessions")
USER_SESSION = os.getenv("SESSION_NAME", "user.session")
BOT_SESSION  = os.getenv("BOT_SESSION_NAME", "bot.session")

TELETHON_SESSION_B64 = os.getenv("TELETHON_SESSION_B64", "").strip()

MIN_DELAY = max(0, int(os.getenv("MIN_DELAY", "1")))
MAX_DELAY = max(MIN_DELAY, int(os.getenv("MAX_DELAY", "15")))
VERBOSE   = os.getenv("VERBOSE", "0").lower() in {"1","true","yes"}
# --- ключевые слова и стоп-слова (жёстко в коде) ---
KEYWORDS: list[str] = [
    # русские
    "анимация", "аниматор", "анимации",
    "моушн", "моушн дизайнер", "графика",
    "видео", "монтаж", "креатор", "ai креатор",
    "2d", "3d", "rig", "rigging", "explainer",
    # английские
    "motion", "ai", "ai creator", "creator",
    "runway", "midjourney", "blender",
    "c4d", "cinema 4d", "after effects", "ae",
]

NEGATIVE_WORDS: list[str] = [
    # посты «я ищу работу/клиентов/сделаю» — отсекаем
    "ищу работу", "ищу заказ", "ищу заказы", "ищу клиентов",
    "исполнитель", "портфолио", "стажировка",
    "готов выполн", "резюме", "выполню", "сделаю",
    "предлагаю услуги",
]

# Админы (через запятую): ADMIN_IDS=111,222
ADMIN_IDS: Set[int] = {
    int(x) for x in os.getenv("ADMIN_IDS", "").replace(" ", "").split(",") if x.isdigit()
}

DEFAULT_REPLY = (
    "Привет!\n\n"
    "Я Белек, AI creator | Motion graphic designer (Астана)\n"
    "7 лет делаю дизайн и анимацию.\n\n"
    "Работал: Sber, Пятёрочка, VK, Coca-Cola, KPMG\n"
    "Стеки: VEO 3, Midjourney, Runway, Kling, Minimax; AE, Blender, C4D\n"
    "Портфолио:\n- https://www.youtube.com/watch?v=WWyDfEKFvPI\n- https://www.behance.net/beleksadakbek\n\n"
    "Готов обсудить задачу и сроки — могу созвониться."
)

KEYWORDS = [s.strip().lower() for s in os.getenv(
    "KEYWORDS",
    "анимация,motion,моушн,2d,3d,видео,монтаж,rig,rigging,explainer,runway,midjourney,blender,c4d,cinema 4d,after effects,ae,ai,ai creator,creator,креатор,моушн дизайнер,аниматор,анимации,графика"
).split(",") if s.strip()]

NEGATIVE_WORDS = [s.strip().lower() for s in os.getenv(
    "NEGATIVE_WORDS",
    "ищу работу,исполнитель,портфолио,стажировка,готов выполн,резюме,ищу заказы,ищу клиентов,выполню,сделаю,предлагаю услуги"
).split(",") if s.strip()]

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "0").lower() in {"1", "true", "yes"} and bool(GEMINI_API_KEY)

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motionbot")

# -------- Состояние --------
state_lock = asyncio.Lock()
STATE_DEFAULT: Dict[str, Any] = {
    "enabled": False,
    "channels": [],
    "reply_text": DEFAULT_REPLY,
    "owner_id": 0
}

def ensure_dirs():
    pathlib.Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(STATE_FILE) or ".").mkdir(parents=True, exist_ok=True)

def load_state() -> Dict[str, Any]:
    ensure_dirs()
    if not os.path.exists(STATE_FILE):
        return dict(STATE_DEFAULT)
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in STATE_DEFAULT.items():
            data.setdefault(k, v)
        return data
    except Exception as e:
        log.warning(f"state load failed: {e}")
        return dict(STATE_DEFAULT)

async def save_state(st: Dict[str, Any]):
    ensure_dirs()
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

def preview(s: str, n=160):
    s = (s or "").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

def norm(s: str) -> str:
    s = (s or "").replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip()

# Нормализация chat_id/канального id: делаем положительным (убираем -100)
def sid(x: int) -> int:
    return abs(int(x))

# -------- Восстановление user.session из ENV (если нужно) --------
def maybe_restore_user_session_from_b64():
    if not TELETHON_SESSION_B64:
        return
    p = pathlib.Path(SESSIONS_DIR) / USER_SESSION
    if p.exists():
        return
    try:
        raw = base64.b64decode(TELETHON_SESSION_B64)
        p.write_bytes(raw)
        log.info(f"User session restored from TELETHON_SESSION_B64 → {p}")
    except Exception as e:
        log.error(f"Restore session failed: {e}")

# -------- Клиенты --------
if not API_ID or not API_HASH:
    raise SystemExit("TELEGRAM_API_ID/TELEGRAM_API_HASH required in .env")
if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN required in .env")

ensure_dirs()
maybe_restore_user_session_from_b64()

user_session_path = str(pathlib.Path(SESSIONS_DIR) / USER_SESSION)
bot_session_path  = str(pathlib.Path(SESSIONS_DIR) / BOT_SESSION)

user_client = TelegramClient(user_session_path, API_ID, API_HASH)
bot_client  = TelegramClient(bot_session_path,  API_ID, API_HASH)

# Gemini init
if GEMINI_ENABLED:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        log.info(f"Gemini ON ({GEMINI_MODEL})")
    except Exception as e:
        GEMINI_ENABLED = False
        log.warning(f"Gemini init failed → disabled: {e}")

# -------- Фильтры --------
HIRE_RX = re.compile(r"(ищу|ищем|нужен|нужна|нужны|требуется|ваканси|в\s*поиск|на\s*проект|задача)", re.I)

def positive_by_keywords(text: str) -> bool:
    t = text.lower()
    if not any(k in t for k in KEYWORDS):
        return False
    if not HIRE_RX.search(t):
        return False
    return True

def negative_by_keywords(text: str) -> bool:
    t = text.lower()
    return any(n in t for n in NEGATIVE_WORDS)

async def gemini_relevant(text: str) -> bool:
    """Строгая классификация через Gemini с JSON-ответом."""
    if not GEMINI_ENABLED:
        return False
    try:
        prompt = (
            "Ты модератор вакансий по motion/2D/3D/анимации.\n"
            "Классифицируй текст как ЗАПРОС ИСПОЛНИТЕЛЯ (клиент ищет специалиста) или НЕТ.\n"
            "Важно: не считать релевантным, если автор сам ищет заказ/работу.\n"
            "Ответ строго в JSON без пояснений, на одной строке:\n"
            '{"relevant": true}  или  {"relevant": false}\n\n'
            f"Текст: \"\"\"{text}\"\"\""
        )
        resp = gemini_model.generate_content(prompt)
        raw = (getattr(resp, "text", None) or "").strip()
        try:
            j = json.loads(raw)
            return bool(j.get("relevant") is True)
        except Exception:
            return bool(re.search(r'"relevant"\s*:\s*true', raw, re.I))
    except Exception as e:
        log.warning(f"Gemini error: {e}")
        return False

# -------- Разрешённые каналы --------
ALLOW_IDS: Set[int] = set()
RESOLVE_LOCK = asyncio.Lock()

def _normalize_channel_link(x: str) -> str:
    x = x.strip()
    if not x:
        return ""
    if x.startswith("@"):
        x = "https://t.me/" + x[1:]
    return x

async def resolve_channels():
    """Резолвит ссылки/юзернеймы из state → ALLOW_IDS (нормализованные id)."""
    global ALLOW_IDS
    async with RESOLVE_LOCK:
        st = load_state()
        channels: List[str] = st.get("channels") or []
        if not channels:
            ALLOW_IDS = set()
            log.warning("CHANNELS пуст — слушаем всё (только логируем chat_id/title).")
            return
        ids: List[int] = []
        for link in channels:
            uname = link
            if link.startswith("https://t.me/"):
                uname = link.split("https://t.me/")[-1].strip("/")
            try:
                ent = await user_client.get_entity(uname)
                ids.append(sid(ent.id))  # ключевой фикс: нормализуем id
                title = getattr(ent, "title", None) or getattr(ent, "username", None) or str(ent.id)
                log.info(f"Resolved: {link} → id={ent.id} | {ent.__class__.__name__} | {title}")
            except Exception as e:
                log.error(f"Resolve failed: {link} → {e}")
        ALLOW_IDS = set(ids)
        log.info(f"Listening IDs: {sorted(ALLOW_IDS)}")

# -------- Монитор новых сообщений --------
@user_client.on(events.NewMessage)
async def watcher(event):
    key = f"{event.chat_id}:{event.id}"
    try:
        st = load_state()
        if not st.get("enabled"):
            return
        if ALLOW_IDS and (sid(event.chat_id) not in ALLOW_IDS):  # фикс сравнения
            return

        raw = event.raw_text or ""
        text = norm(raw)
        chat_title = getattr(getattr(event, "chat", None), "title", None)
        log.info(f"[{key}] New in '{chat_title or event.chat_id}': {preview(raw)}")

        ok = False
        if positive_by_keywords(text) and not negative_by_keywords(text):
            ok = True
        else:
            log.info(f"[{key}] keywords unsure → Gemini")
            ok = await gemini_relevant(text)

        if not ok:
            log.info(f"[{key}] Decision: SKIP")
            return

        sender = await event.get_sender()
        if sender and getattr(sender, "is_self", False):
            return
        if sender and getattr(sender, "bot", False):
            return

        m = re.search(r"@([A-Za-z0-9_]{4,32})", raw) or re.search(r"t\.me/([A-Za-z0-9_]{4,32})", raw, re.I)
        if m:
            target = f"@{m.group(1)}"
        elif getattr(sender, "username", None):
            target = f"@{sender.username}"
        else:
            target = getattr(sender, "id", None)

        if not target:
            log.info(f"[{key}] No target → SKIP")
            return

        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        log.info(f"[{key}] Sleep {delay:.1f}s")
        await asyncio.sleep(delay)

        try:
            await user_client.send_message(target, st.get("reply_text") or DEFAULT_REPLY)
            log.info(f"[{key}] SENT → {target}")
            if st.get("owner_id"):
                src = chat_title or str(event.chat_id)
                who = target if isinstance(target, str) else f"id:{target}"
                asyncio.create_task(
                    bot_client.send_message(st["owner_id"], f"✅ Отклик отправлен {who} (источник: {src})")
                )
        except FloodWaitError as e:
            log.warning(f"[{key}] FloodWait {e.seconds}s")
        except (UserPrivacyRestrictedError, UserIsBlockedError):
            log.warning(f"[{key}] privacy/blocked → {target}")
        except (UsernameInvalidError, PeerIdInvalidError, ChatWriteForbiddenError) as e:
            log.warning(f"[{key}] cannot write → {target}: {e}")
        except Exception as e:
            log.error(f"[{key}] send failed → {target}: {e}")

    except Exception as e:
        log.exception(f"[{key}] handler error: {e}")

# -------- Команды бота --------
def is_owner(st: Dict[str, Any], user_id: int) -> bool:
    owner = int(st.get("owner_id") or 0)
    # владельцем считается первый /start, но админы тоже могут управлять
    return (user_id in ADMIN_IDS) or owner == 0 or owner == user_id

@bot_client.on(events.NewMessage(pattern=r'^/start$'))
async def cmd_start(event):
    st = load_state()
    if st.get("owner_id") in (0, None):
        async with state_lock:
            st["owner_id"] = event.sender_id
            await save_state(st)
    txt = (
        "Привет! Я управляю сканером заказов.\n\n"
        "Команды:\n"
        "/on — включить\n"
        "/off — выключить\n"
        "/add_channel <ссылки или @username через пробел/переносы>\n"
        "/list_channels — список\n"
        "/remove_channel <номер>\n"
        "/set_reply <текст> — шаблон отклика\n"
        "/status — состояние\n"
        "/whoami — показать ваш user_id\n"
    )
    await event.respond(txt)

@bot_client.on(events.NewMessage(pattern=r'^/whoami$'))
async def cmd_whoami(event):
    await event.respond(f"Ваш user_id: {event.sender_id}")

@bot_client.on(events.NewMessage(pattern=r'^/on$'))
async def cmd_on(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["enabled"] = True
        await save_state(st)
    await event.respond("Мониторинг включён ✅")

@bot_client.on(events.NewMessage(pattern=r'^/off$'))
async def cmd_off(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["enabled"] = False
        await save_state(st)
    await event.respond("Мониторинг выключен ⏸️")

def _split_links(s: str) -> List[str]:
    # Делим по пробелам, переводам строк и запятым; фильтруем только @юзернеймы и t.me/*
    items = re.split(r"[\s,]+", s.strip())
    out: List[str] = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        if it.startswith("@") or it.startswith("https://t.me/"):
            out.append(_normalize_channel_link(it))
        else:
            if re.fullmatch(r"[A-Za-z0-9_]{4,32}", it):  # голое имя канала
                out.append("https://t.me/" + it)
    return out

@bot_client.on(events.NewMessage(pattern=r'^/add_channel\s+([\s\S]+)$'))
async def cmd_add_channel(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    raw = event.pattern_match.group(1)
    links = _split_links(raw)
    if not links:
        return await event.respond("Не нашёл ни одной ссылки/юзернейма.")
    added = []
    async with state_lock:
        for link in links:
            if link and link not in st["channels"]:
                st["channels"].append(link)
                added.append(link)
        await save_state(st)
    if added:
        await event.respond("Добавил:\n" + "\n".join(f"- {x}" for x in added) + "\nРезолвим…")
    else:
        await event.respond("Все указанные каналы уже были в списке.")
    await resolve_channels()

@bot_client.on(events.NewMessage(pattern=r'^/list_channels$'))
async def cmd_list_channels(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    if not st["channels"]:
        return await event.respond("Список пуст.")
    body = "\n".join(f"{i+1}. {c}" for i, c in enumerate(st["channels"]))
    await event.respond(f"Каналы:\n{body}")

@bot_client.on(events.NewMessage(pattern=r'^/remove_channel\s+(\d+)$'))
async def cmd_remove_channel(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    idx = int(event.pattern_match.group(1)) - 1
    if idx < 0 or idx >= len(st["channels"]):
        return await event.respond("Некорректный номер.")
    async with state_lock:
        removed = st["channels"].pop(idx)
        await save_state(st)
    await event.respond(f"Удалил: {removed}")
    await resolve_channels()

@bot_client.on(events.NewMessage(pattern=r'^/set_reply\s+([\s\S]+)$'))
async def cmd_set_reply(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    txt = event.pattern_match.group(1).strip()
    async with state_lock:
        st["reply_text"] = txt or DEFAULT_REPLY
        await save_state(st)
    await event.respond("Шаблон отклика обновлён ✅")

@bot_client.on(events.NewMessage(pattern=r'^/status$'))
async def cmd_status(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    ch = "\n".join(f"- {c}" for c in st["channels"]) or "пусто"
    await event.respond(
        f"Включено: {st['enabled']}\n"
        f"Каналы:\n{ch}\n"
        f"Задержка: {MIN_DELAY}-{MAX_DELAY} c\n"
        f"Gemini: {'ON' if GEMINI_ENABLED else 'OFF'}\n"
        f"Админы: {', '.join(map(str, ADMIN_IDS)) or 'нет'}"
    )

# -------- Старт --------
async def main():
    log.info("=== CONFIG ===")
    log.info(f"Delay: {MIN_DELAY}-{MAX_DELAY}s | VERBOSE={VERBOSE}")
    log.info(f"Sessions dir: {SESSIONS_DIR} | USER={USER_SESSION} | BOT={BOT_SESSION}")
    log.info(f"State: {STATE_FILE}")
    log.info(f"Gemini: {'ON' if GEMINI_ENABLED else 'OFF'}")
    await bot_client.start(bot_token=BOT_TOKEN)
    await user_client.start()
    await resolve_channels()
    log.info("Both clients started. Waiting for events...")
    await asyncio.gather(
        bot_client.run_until_disconnected(),
        user_client.run_until_disconnected()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
# === DEBUG COMMANDS ===

@bot_client.on(events.NewMessage(pattern=r'^/debug_status$'))
async def cmd_debug_status(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    await event.respond(
        "DEBUG:\n"
        f"- enabled={st.get('enabled')}\n"
        f"- ALLOW_IDS={sorted(ALLOW_IDS) if ALLOW_IDS else 'ALL'}\n"
        f"- reply_len={len((st.get('reply_text') or DEFAULT_REPLY))}\n"
    )

@bot_client.on(events.NewMessage(pattern=r'^/debug_test_send\s+(\S+)$'))
async def cmd_debug_test_send(event):
    """Пробная отправка текущего шаблона отклика на указанный @username или числовой id."""
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    target = event.pattern_match.group(1)
    try:
        if target.startswith("@"):
            to = target
        elif target.isdigit():
            to = int(target)
        else:
            return await event.respond("Формат: /debug_test_send @username ИЛИ /debug_test_send <user_id>")
        await user_client.send_message(to, st.get("reply_text") or DEFAULT_REPLY)
        await event.respond(f"✅ Тестовое ЛС отправлено → {to}")
    except Exception as e:
        await event.respond(f"❌ Ошибка отправки: {e}")

@bot_client.on(events.NewMessage(pattern=r'^/debug_test_text\s+([\s\S]+)$'))
async def cmd_debug_test_text(event):
    """Прогон строки через ключи+негативы и Gemini, вернуть вердикт."""
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    text = norm(event.pattern_match.group(1))
    pos = positive_by_keywords(text)
    neg = negative_by_keywords(text)
    gmi = await gemini_relevant(text)
    verdict = (pos and not neg) or (not (pos and not neg) and gmi)
    await event.respond(
        "Проверка текста:\n"
        f"- positive_by_keywords={pos}\n"
        f"- negative_by_keywords={neg}\n"
        f"- gemini={gmi}\n"
        f"- FINAL={verdict}"
    )# -*- coding: utf-8 -*-
"""
Управляющий бот + сканер каналов на Telethon.
- bot_client: Telegram bot по BOT_TOKEN (команды /start /on /off /add_channel /list_channels /remove_channel /set_reply /status /whoami)
- user_client: обычный аккаунт (api_id/api_hash) слушает каналы и шлёт ЛС контактам
- Ключевые слова + fallback на Gemini (google-generativeai)
"""

import os, re, json, random, asyncio, logging, base64, pathlib
from typing import Dict, Any, List, Set

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors.rpcerrorlist import (
    FloodWaitError, PeerIdInvalidError, UsernameInvalidError,
    ChatWriteForbiddenError, UserIsBlockedError, UserPrivacyRestrictedError
)

# -------- Gemini --------
import google.generativeai as genai

# -------- Конфиг/ENV --------
load_dotenv()

BOT_TOKEN  = os.getenv("BOT_TOKEN", "").strip()
API_ID     = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH   = os.getenv("TELEGRAM_API_HASH", "").strip()

STATE_FILE   = os.getenv("STATE_FILE", "./data/state.json")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "./.sessions")
USER_SESSION = os.getenv("SESSION_NAME", "user.session")
BOT_SESSION  = os.getenv("BOT_SESSION_NAME", "bot.session")

TELETHON_SESSION_B64 = os.getenv("TELETHON_SESSION_B64", "").strip()

MIN_DELAY = max(0, int(os.getenv("MIN_DELAY", "1")))
MAX_DELAY = max(MIN_DELAY, int(os.getenv("MAX_DELAY", "15")))
VERBOSE   = os.getenv("VERBOSE", "0").lower() in {"1","true","yes"}

# Админы (через запятую): ADMIN_IDS=111,222
ADMIN_IDS: Set[int] = {
    int(x) for x in os.getenv("ADMIN_IDS", "").replace(" ", "").split(",") if x.isdigit()
}

DEFAULT_REPLY = (
    "Привет!\n\n"
    "Я Белек, AI creator | Motion graphic designer (Астана)\n"
    "7 лет делаю дизайн и анимацию.\n\n"
    "Работал: Sber, Пятёрочка, VK, Coca-Cola, KPMG\n"
    "Стеки: VEO 3, Midjourney, Runway, Kling, Minimax; AE, Blender, C4D\n"
    "Портфолио:\n- https://www.youtube.com/watch?v=WWyDfEKFvPI\n- https://www.behance.net/beleksadakbek\n\n"
    "Готов обсудить задачу и сроки — могу созвониться."
)

KEYWORDS = [s.strip().lower() for s in os.getenv(
    "KEYWORDS",
    "анимация,motion,моушн,2d,3d,видео,монтаж,rig,rigging,explainer,runway,midjourney,blender,c4d,cinema 4d,after effects,ae,ai,ai creator,creator,креатор,моушн дизайнер,аниматор,анимации,графика"
).split(",") if s.strip()]

NEGATIVE_WORDS = [s.strip().lower() for s in os.getenv(
    "NEGATIVE_WORDS",
    "ищу работу,исполнитель,портфолио,стажировка,готов выполн,резюме,ищу заказы,ищу клиентов,выполню,сделаю,предлагаю услуги"
).split(",") if s.strip()]

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "0").lower() in {"1", "true", "yes"} and bool(GEMINI_API_KEY)

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motionbot")

# -------- Состояние --------
state_lock = asyncio.Lock()
STATE_DEFAULT: Dict[str, Any] = {
    "enabled": False,
    "channels": [],
    "reply_text": DEFAULT_REPLY,
    "owner_id": 0
}

def ensure_dirs():
    pathlib.Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(STATE_FILE) or ".").mkdir(parents=True, exist_ok=True)

def load_state() -> Dict[str, Any]:
    ensure_dirs()
    if not os.path.exists(STATE_FILE):
        return dict(STATE_DEFAULT)
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in STATE_DEFAULT.items():
            data.setdefault(k, v)
        return data
    except Exception as e:
        log.warning(f"state load failed: {e}")
        return dict(STATE_DEFAULT)

async def save_state(st: Dict[str, Any]):
    ensure_dirs()
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

def preview(s: str, n=160):
    s = (s or "").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

def norm(s: str) -> str:
    s = (s or "").replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip()

# Нормализация chat_id/канального id: делаем положительным (убираем -100)
def sid(x: int) -> int:
    return abs(int(x))

# -------- Восстановление user.session из ENV (если нужно) --------
def maybe_restore_user_session_from_b64():
    if not TELETHON_SESSION_B64:
        return
    p = pathlib.Path(SESSIONS_DIR) / USER_SESSION
    if p.exists():
        return
    try:
        raw = base64.b64decode(TELETHON_SESSION_B64)
        p.write_bytes(raw)
        log.info(f"User session restored from TELETHON_SESSION_B64 → {p}")
    except Exception as e:
        log.error(f"Restore session failed: {e}")

# -------- Клиенты --------
if not API_ID or not API_HASH:
    raise SystemExit("TELEGRAM_API_ID/TELEGRAM_API_HASH required in .env")
if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN required in .env")

ensure_dirs()
maybe_restore_user_session_from_b64()

user_session_path = str(pathlib.Path(SESSIONS_DIR) / USER_SESSION)
bot_session_path  = str(pathlib.Path(SESSIONS_DIR) / BOT_SESSION)

user_client = TelegramClient(user_session_path, API_ID, API_HASH)
bot_client  = TelegramClient(bot_session_path,  API_ID, API_HASH)

# Gemini init
if GEMINI_ENABLED:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        log.info(f"Gemini ON ({GEMINI_MODEL})")
    except Exception as e:
        GEMINI_ENABLED = False
        log.warning(f"Gemini init failed → disabled: {e}")

# -------- Фильтры --------
HIRE_RX = re.compile(r"(ищу|ищем|нужен|нужна|нужны|требуется|ваканси|в\s*поиск|на\s*проект|задача)", re.I)

def positive_by_keywords(text: str) -> bool:
    t = text.lower()
    if not any(k in t for k in KEYWORDS):
        return False
    if not HIRE_RX.search(t):
        return False
    return True

def negative_by_keywords(text: str) -> bool:
    t = text.lower()
    return any(n in t for n in NEGATIVE_WORDS)

async def gemini_relevant(text: str) -> bool:
    """Строгая классификация через Gemini с JSON-ответом."""
    if not GEMINI_ENABLED:
        return False
    try:
        prompt = (
            "Ты модератор вакансий по motion/2D/3D/анимации.\n"
            "Классифицируй текст как ЗАПРОС ИСПОЛНИТЕЛЯ (клиент ищет специалиста) или НЕТ.\n"
            "Важно: не считать релевантным, если автор сам ищет заказ/работу.\n"
            "Ответ строго в JSON без пояснений, на одной строке:\n"
            '{"relevant": true}  или  {"relevant": false}\n\n'
            f"Текст: \"\"\"{text}\"\"\""
        )
        resp = gemini_model.generate_content(prompt)
        raw = (getattr(resp, "text", None) or "").strip()
        try:
            j = json.loads(raw)
            return bool(j.get("relevant") is True)
        except Exception:
            return bool(re.search(r'"relevant"\s*:\s*true', raw, re.I))
    except Exception as e:
        log.warning(f"Gemini error: {e}")
        return False

# -------- Разрешённые каналы --------
ALLOW_IDS: Set[int] = set()
RESOLVE_LOCK = asyncio.Lock()

def _normalize_channel_link(x: str) -> str:
    x = x.strip()
    if not x:
        return ""
    if x.startswith("@"):
        x = "https://t.me/" + x[1:]
    return x

async def resolve_channels():
    """Резолвит ссылки/юзернеймы из state → ALLOW_IDS (нормализованные id)."""
    global ALLOW_IDS
    async with RESOLVE_LOCK:
        st = load_state()
        channels: List[str] = st.get("channels") or []
        if not channels:
            ALLOW_IDS = set()
            log.warning("CHANNELS пуст — слушаем всё (только логируем chat_id/title).")
            return
        ids: List[int] = []
        for link in channels:
            uname = link
            if link.startswith("https://t.me/"):
                uname = link.split("https://t.me/")[-1].strip("/")
            try:
                ent = await user_client.get_entity(uname)
                ids.append(sid(ent.id))  # ключевой фикс: нормализуем id
                title = getattr(ent, "title", None) or getattr(ent, "username", None) or str(ent.id)
                log.info(f"Resolved: {link} → id={ent.id} | {ent.__class__.__name__} | {title}")
            except Exception as e:
                log.error(f"Resolve failed: {link} → {e}")
        ALLOW_IDS = set(ids)
        log.info(f"Listening IDs: {sorted(ALLOW_IDS)}")

# -------- Монитор новых сообщений --------
@user_client.on(events.NewMessage)
async def watcher(event):
    key = f"{event.chat_id}:{event.id}"
    try:
        st = load_state()
        if not st.get("enabled"):
            return
        if ALLOW_IDS and (sid(event.chat_id) not in ALLOW_IDS):  # фикс сравнения
            return

        raw = event.raw_text or ""
        text = norm(raw)
        chat_title = getattr(getattr(event, "chat", None), "title", None)
        log.info(f"[{key}] New in '{chat_title or event.chat_id}': {preview(raw)}")

        ok = False
        if positive_by_keywords(text) and not negative_by_keywords(text):
            ok = True
        else:
            log.info(f"[{key}] keywords unsure → Gemini")
            ok = await gemini_relevant(text)

        if not ok:
            log.info(f"[{key}] Decision: SKIP")
            return

        sender = await event.get_sender()
        if sender and getattr(sender, "is_self", False):
            return
        if sender and getattr(sender, "bot", False):
            return

        m = re.search(r"@([A-Za-z0-9_]{4,32})", raw) or re.search(r"t\.me/([A-Za-z0-9_]{4,32})", raw, re.I)
        if m:
            target = f"@{m.group(1)}"
        elif getattr(sender, "username", None):
            target = f"@{sender.username}"
        else:
            target = getattr(sender, "id", None)

        if not target:
            log.info(f"[{key}] No target → SKIP")
            return

        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        log.info(f"[{key}] Sleep {delay:.1f}s")
        await asyncio.sleep(delay)

        try:
            await user_client.send_message(target, st.get("reply_text") or DEFAULT_REPLY)
            log.info(f"[{key}] SENT → {target}")
            if st.get("owner_id"):
                src = chat_title or str(event.chat_id)
                who = target if isinstance(target, str) else f"id:{target}"
                asyncio.create_task(
                    bot_client.send_message(st["owner_id"], f"✅ Отклик отправлен {who} (источник: {src})")
                )
        except FloodWaitError as e:
            log.warning(f"[{key}] FloodWait {e.seconds}s")
        except (UserPrivacyRestrictedError, UserIsBlockedError):
            log.warning(f"[{key}] privacy/blocked → {target}")
        except (UsernameInvalidError, PeerIdInvalidError, ChatWriteForbiddenError) as e:
            log.warning(f"[{key}] cannot write → {target}: {e}")
        except Exception as e:
            log.error(f"[{key}] send failed → {target}: {e}")

    except Exception as e:
        log.exception(f"[{key}] handler error: {e}")

# -------- Команды бота --------
def is_owner(st: Dict[str, Any], user_id: int) -> bool:
    owner = int(st.get("owner_id") or 0)
    # владельцем считается первый /start, но админы тоже могут управлять
    return (user_id in ADMIN_IDS) or owner == 0 or owner == user_id

@bot_client.on(events.NewMessage(pattern=r'^/start$'))
async def cmd_start(event):
    st = load_state()
    if st.get("owner_id") in (0, None):
        async with state_lock:
            st["owner_id"] = event.sender_id
            await save_state(st)
    txt = (
        "Привет! Я управляю сканером заказов.\n\n"
        "Команды:\n"
        "/on — включить\n"
        "/off — выключить\n"
        "/add_channel <ссылки или @username через пробел/переносы>\n"
        "/list_channels — список\n"
        "/remove_channel <номер>\n"
        "/set_reply <текст> — шаблон отклика\n"
        "/status — состояние\n"
        "/whoami — показать ваш user_id\n"
    )
    await event.respond(txt)

@bot_client.on(events.NewMessage(pattern=r'^/whoami$'))
async def cmd_whoami(event):
    await event.respond(f"Ваш user_id: {event.sender_id}")

@bot_client.on(events.NewMessage(pattern=r'^/on$'))
async def cmd_on(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["enabled"] = True
        await save_state(st)
    await event.respond("Мониторинг включён ✅")

@bot_client.on(events.NewMessage(pattern=r'^/off$'))
async def cmd_off(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["enabled"] = False
        await save_state(st)
    await event.respond("Мониторинг выключен ⏸️")

def _split_links(s: str) -> List[str]:
    # Делим по пробелам, переводам строк и запятым; фильтруем только @юзернеймы и t.me/*
    items = re.split(r"[\s,]+", s.strip())
    out: List[str] = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        if it.startswith("@") or it.startswith("https://t.me/"):
            out.append(_normalize_channel_link(it))
        else:
            if re.fullmatch(r"[A-Za-z0-9_]{4,32}", it):  # голое имя канала
                out.append("https://t.me/" + it)
    return out

@bot_client.on(events.NewMessage(pattern=r'^/add_channel\s+([\s\S]+)$'))
async def cmd_add_channel(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    raw = event.pattern_match.group(1)
    links = _split_links(raw)
    if not links:
        return await event.respond("Не нашёл ни одной ссылки/юзернейма.")
    added = []
    async with state_lock:
        for link in links:
            if link and link not in st["channels"]:
                st["channels"].append(link)
                added.append(link)
        await save_state(st)
    if added:
        await event.respond("Добавил:\n" + "\n".join(f"- {x}" for x in added) + "\nРезолвим…")
    else:
        await event.respond("Все указанные каналы уже были в списке.")
    await resolve_channels()

@bot_client.on(events.NewMessage(pattern=r'^/list_channels$'))
async def cmd_list_channels(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    if not st["channels"]:
        return await event.respond("Список пуст.")
    body = "\n".join(f"{i+1}. {c}" for i, c in enumerate(st["channels"]))
    await event.respond(f"Каналы:\n{body}")

@bot_client.on(events.NewMessage(pattern=r'^/remove_channel\s+(\d+)$'))
async def cmd_remove_channel(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    idx = int(event.pattern_match.group(1)) - 1
    if idx < 0 or idx >= len(st["channels"]):
        return await event.respond("Некорректный номер.")
    async with state_lock:
        removed = st["channels"].pop(idx)
        await save_state(st)
    await event.respond(f"Удалил: {removed}")
    await resolve_channels()

@bot_client.on(events.NewMessage(pattern=r'^/set_reply\s+([\s\S]+)$'))
async def cmd_set_reply(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    txt = event.pattern_match.group(1).strip()
    async with state_lock:
        st["reply_text"] = txt or DEFAULT_REPLY
        await save_state(st)
    await event.respond("Шаблон отклика обновлён ✅")

@bot_client.on(events.NewMessage(pattern=r'^/status$'))
async def cmd_status(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    ch = "\n".join(f"- {c}" for c in st["channels"]) or "пусто"
    await event.respond(
        f"Включено: {st['enabled']}\n"
        f"Каналы:\n{ch}\n"
        f"Задержка: {MIN_DELAY}-{MAX_DELAY} c\n"
        f"Gemini: {'ON' if GEMINI_ENABLED else 'OFF'}\n"
        f"Админы: {', '.join(map(str, ADMIN_IDS)) or 'нет'}"
    )

# -------- Старт --------
async def main():
    log.info("=== CONFIG ===")
    log.info(f"Delay: {MIN_DELAY}-{MAX_DELAY}s | VERBOSE={VERBOSE}")
    log.info(f"Sessions dir: {SESSIONS_DIR} | USER={USER_SESSION} | BOT={BOT_SESSION}")
    log.info(f"State: {STATE_FILE}")
    log.info(f"Gemini: {'ON' if GEMINI_ENABLED else 'OFF'}")
    await bot_client.start(bot_token=BOT_TOKEN)
    await user_client.start()
    await resolve_channels()
    log.info("Both clients started. Waiting for events...")
    await asyncio.gather(
        bot_client.run_until_disconnected(),
        user_client.run_until_disconnected()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
# === DEBUG COMMANDS ===

@bot_client.on(events.NewMessage(pattern=r'^/debug_status$'))
async def cmd_debug_status(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    await event.respond(
        "DEBUG:\n"
        f"- enabled={st.get('enabled')}\n"
        f"- ALLOW_IDS={sorted(ALLOW_IDS) if ALLOW_IDS else 'ALL'}\n"
        f"- reply_len={len((st.get('reply_text') or DEFAULT_REPLY))}\n"
    )

@bot_client.on(events.NewMessage(pattern=r'^/debug_test_send\s+(\S+)$'))
async def cmd_debug_test_send(event):
    """Пробная отправка текущего шаблона отклика на указанный @username или числовой id."""
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    target = event.pattern_match.group(1)
    try:
        if target.startswith("@"):
            to = target
        elif target.isdigit():
            to = int(target)
        else:
            return await event.respond("Формат: /debug_test_send @username ИЛИ /debug_test_send <user_id>")
        await user_client.send_message(to, st.get("reply_text") or DEFAULT_REPLY)
        await event.respond(f"✅ Тестовое ЛС отправлено → {to}")
    except Exception as e:
        await event.respond(f"❌ Ошибка отправки: {e}")

@bot_client.on(events.NewMessage(pattern=r'^/debug_test_text\s+([\s\S]+)$'))
async def cmd_debug_test_text(event):
    """Прогон строки через ключи+негативы и Gemini, вернуть вердикт."""
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    text = norm(event.pattern_match.group(1))
    pos = positive_by_keywords(text)
    neg = negative_by_keywords(text)
    gmi = await gemini_relevant(text)
    verdict = (pos and not neg) or (not (pos and not neg) and gmi)
    await event.respond(
        "Проверка текста:\n"
        f"- positive_by_keywords={pos}\n"
        f"- negative_by_keywords={neg}\n"
        f"- gemini={gmi}\n"
        f"- FINAL={verdict}"
    )

