# -*- coding: utf-8 -*-
"""
Управление и мониторинг — всё на Telethon:
- bot_client: подключается по BOT_TOKEN, принимает команды /start /on /off /add_channel /list_channels /remove_channel /set_reply /status
- user_client: подключается как обычный аккаунт, мониторит каналы и шлёт ЛС
- ключевые слова + fallback на Gemini (опционально)
"""

import os, re, json, random, asyncio, logging, base64, pathlib
from typing import Dict, Any, List

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors.rpcerrorlist import (
    FloodWaitError, PeerIdInvalidError, UsernameInvalidError,
    ChatWriteForbiddenError, UserIsBlockedError, UserPrivacyRestrictedError
)

# === Gemini (опц.) ===
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ------------------ Конфиг ------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "").strip()

STATE_FILE   = os.getenv("STATE_FILE", "./data/state.json")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "./.sessions")
USER_SESSION = os.getenv("SESSION_NAME", "user.session")     # юзер-сессия
BOT_SESSION  = os.getenv("BOT_SESSION_NAME", "bot.session")  # сессия бота (не обязательно)

# Восстановление юзер-сессии из ENV (для Pella)
TELETHON_SESSION_B64 = os.getenv("TELETHON_SESSION_B64", "").strip()

MIN_DELAY = max(0, int(os.getenv("MIN_DELAY", "1")))
MAX_DELAY = max(MIN_DELAY, int(os.getenv("MAX_DELAY", "15")))
VERBOSE   = os.getenv("VERBOSE", "0").lower() in {"1","true","yes"}

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
    "анимация,motion,моушн,2d,3d,видео,монтаж,rig,rigging,explainer,runway,midjourney,blender,c4d,cinema 4d,after effects,ae"
).split(",") if s.strip()]

NEGATIVE_WORDS = [s.strip().lower() for s in os.getenv(
    "NEGATIVE_WORDS",
    "ищу работу,исполнитель,портфолио,стажировка,готов выполн,резюме"
).split(",") if s.strip()]

GEMINI_ENABLED = os.getenv("GEMINI_ENABLED","0").lower() in {"1","true","yes"} and GEMINI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY","").strip())
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","").strip()

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motionbot")

# ------------------ Утилиты состояния ------------------
state_lock = asyncio.Lock()
STATE_DEFAULT: Dict[str, Any] = {
    "enabled": False,
    "channels": [],             # ссылки/юзернеймы каналов
    "reply_text": DEFAULT_REPLY,
    "owner_id": 0               # сюда слать отчёты
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
        for k,v in STATE_DEFAULT.items():
            data.setdefault(k,v)
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
    s = (s or "").replace("\n"," ")
    return (s[:n] + "…") if len(s) > n else s

def norm(s: str) -> str:
    s = (s or "").replace("ё","е")
    return re.sub(r"\s+"," ",s).strip()

# ------------------ восстановление user.session из ENV ------------------
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

# ------------------ Клиенты ------------------
if not API_ID or not API_HASH:
    raise SystemExit("TELEGRAM_API_ID/TELEGRAM_API_HASH required in .env")
if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN required in .env")

ensure_dirs()
maybe_restore_user_session_from_b64()

user_session_path = str(pathlib.Path(SESSIONS_DIR)/USER_SESSION)
bot_session_path  = str(pathlib.Path(SESSIONS_DIR)/BOT_SESSION)

user_client = TelegramClient(user_session_path, API_ID, API_HASH)          # юзер
bot_client  = TelegramClient(bot_session_path,  API_ID, API_HASH)          # бот

# Gemini init
if GEMINI_ENABLED:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        log.info(f"Gemini ON ({GEMINI_MODEL})")
    except Exception as e:
        GEMINI_ENABLED = False
        log.warning(f"Gemini init failed → disabled: {e}")

# ------------------ Фильтры ------------------
def positive_by_keywords(text: str) -> bool:
    t = text.lower()
    if not any(k in t for k in KEYWORDS):
        return False
    if not re.search(r"(ищу|ищем|нужен|нужна|нужны|требуется|ваканси|в\s*поиск|на\s*проект|задача)", t):
        return False
    return True

def negative_by_keywords(text: str) -> bool:
    t = text.lower()
    return any(n in t for n in NEGATIVE_WORDS)

async def gemini_relevant(text: str) -> bool:
    if not GEMINI_ENABLED:
        return False
    try:
        prompt = (
            "Classify Telegram posts in RU/EN as HIRE_REQUEST or NOT for motion/2D/3D/animation jobs.\n"
            "Return ONLY TRUE or FALSE.\n\n"
            f"Text:\n{text}"
        )
        resp = gemini_model.generate_content(prompt)
        out = (resp.text or "").strip().upper()
        return "TRUE" in out and "FALSE" not in out
    except Exception as e:
        log.warning(f"Gemini error: {e}")
        return False

# ------------------ Разрешённые каналы ------------------
ALLOW_IDS: set[int] = set()
RESOLVE_LOCK = asyncio.Lock()

async def resolve_channels():
    global ALLOW_IDS
    async with RESOLVE_LOCK:
        st = load_state()
        channels = st.get("channels") or []
        if not channels:
            ALLOW_IDS = set()
            log.warning("CHANNELS пуст — слушаем всё (только логируем chat_id/title).")
            return
        ids = []
        for link in channels:
            uname = link
            if link.startswith("https://t.me/"):
                uname = link.split("https://t.me/")[-1].strip("/")
            try:
                ent = await user_client.get_entity(uname)
                ids.append(ent.id)
                title = getattr(ent, "title", None) or getattr(ent, "username", None) or str(ent.id)
                log.info(f"Resolved: {link} → id={ent.id} | {ent.__class__.__name__} | {title}")
            except Exception as e:
                log.error(f"Resolve failed: {link} → {e}")
        ALLOW_IDS = set(ids)
        log.info(f"Listening IDs: {sorted(ALLOW_IDS)}")

# ------------------ Мониторинг новых сообщений (юзер-клиент) ------------------
@user_client.on(events.NewMessage)
async def watcher(event):
    key = f"{event.chat_id}:{event.id}"
    try:
        st = load_state()
        if not st.get("enabled"):
            return
        if ALLOW_IDS and (event.chat_id not in ALLOW_IDS):
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
                asyncio.create_task(bot_client.send_message(st["owner_id"], f"✅ Отклик отправлен {who} (источник: {src})"))
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

# ------------------ Команды бота (bot_client) ------------------
def is_owner(st: Dict[str, Any], user_id: int) -> bool:
    owner = int(st.get("owner_id") or 0)
    return owner == 0 or owner == user_id  # 0 → первый /start назначит владельца

@bot_client.on(events.NewMessage(pattern=r'^/start'))
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
        "/add_channel <ссылка или @username>\n"
        "/list_channels — список\n"
        "/remove_channel <номер>\n"
        "/set_reply <текст> — шаблон отклика\n"
        "/status — состояние\n"
    )
    await event.respond(txt)

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

@bot_client.on(events.NewMessage(pattern=r'^/add_channel\s+(.+)$'))
async def cmd_add_channel(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    link = event.pattern_match.group(1).strip()
    if link.startswith("@"):
        link = "https://t.me/" + link[1:]
    async with state_lock:
        if link not in st["channels"]:
            st["channels"].append(link)
            await save_state(st)
    await event.respond(f"Добавил: {link}\nРезолвим…")
    await resolve_channels()

@bot_client.on(events.NewMessage(pattern=r'^/list_channels$'))
async def cmd_list_channels(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    if not st["channels"]:
        return await event.respond("Список пуст.")
    body = "\n".join(f"{i+1}. {c}" for i,c in enumerate(st["channels"]))
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
    txt = event.pattern_match.group(1)
    async with state_lock:
        st["reply_text"] = txt
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
        f"Gemini: {'ON' if GEMINI_ENABLED else 'OFF'}"
    )

# ------------------ Старт обоих клиентов ------------------
async def main():
    log.info("=== CONFIG ===")
    log.info(f"Delay: {MIN_DELAY}-{MAX_DELAY}s | VERBOSE={VERBOSE}")
    log.info(f"Sessions dir: {SESSIONS_DIR} | USER={USER_SESSION} | BOT={BOT_SESSION}")
    log.info(f"State: {STATE_FILE}")
    log.info(f"Gemini: {'ON' if GEMINI_ENABLED else 'OFF'}")
    # стартуем бота и юзера
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
