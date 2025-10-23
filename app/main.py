# -*- coding: utf-8 -*-
"""
MotionHunter — мониторинг телеграм-каналов и авто-отклики.
- bot_client (Telethon): бот по BOT_TOKEN, принимает команды управления
- user_client (Telethon): личный аккаунт, слушает каналы и пишет в ЛС
Логика: ключевые слова + глаголы запроса → при сомнении OpenAI (YES/NO).
Есть аудит, /tap режим, /tail и /show для диагностики. Подписка на каналы.
"""

import os
import re
import json
import random
import asyncio
import logging
import base64
import pathlib
from typing import Dict, Any, List, Iterable, Optional

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.errors.rpcerrorlist import (
    PeerIdInvalidError,
    UsernameInvalidError,
    ChatWriteForbiddenError,
    UserIsBlockedError,
    UserPrivacyRestrictedError,
    UserAlreadyParticipantError,
)
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.functions.messages import ImportChatInviteRequest
from telethon.utils import get_peer_id

# ===== OpenAI (fallback проверка) =====
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_AVAILABLE = False

# ================== Конфиг ==================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "").strip()

STATE_FILE = os.getenv("STATE_FILE", "./data/state.json")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "./.sessions")
USER_SESSION = os.getenv("SESSION_NAME", "user.session")
BOT_SESSION = os.getenv("BOT_SESSION_NAME", "bot.session")

# опционально: восстановление user.session из base64
TELETHON_SESSION_B64 = os.getenv("TELETHON_SESSION_B64", "").strip()

MIN_DELAY = max(0, int(os.getenv("MIN_DELAY", "1")))
MAX_DELAY = max(MIN_DELAY, int(os.getenv("MAX_DELAY", "15")))
VERBOSE = os.getenv("VERBOSE", "0").lower() in {"1", "true", "yes"}

DEFAULT_REPLY = (
    "Привет!\n\n"
    "Я Белек, AI creator | Motion graphic designer (Астана)\n"
    "7 лет занимаюсь дизайном и анимацией.\n\n"
    "Работал с компаниями: Sber, Пятёрочка, VK, Coca-Cola, KPMG\n"
    "Стеки: VEO 3, Midjourney, Runway, Kling, Minimax; After Effects, Blender, C4D\n"
    "Портфолио:\n"
    "- https://www.youtube.com/watch?v=WWyDfEKFvPI\n"
    "- https://www.behance.net/beleksadakbek\n\n"
    "Готов обсудить задачу и сроки — могу созвониться."
)

# -------- Словари (в коде, редактируются командами) --------
KEYWORDS = [
    # исходный набор
    "ищем специалиста 2d", "ищем специалиста 3d",
    "ai creator", "ai креатор", "ai криейтор",
    "ищу моушн дизайнера", "в поиске моушн дизайнера",
    "проектная задача", "2d аниматор",
    "moho", "toon boom специалист",
    "after effects", "монтажёр", "монтажер", "удаленная работа",
    "ии контент", "ии специалист",
    # ai-видео / ролики
    "ии ролик", "ai ролик", "ai видео", "генерация видео",
    # расширения/синонимы
    "motion designer", "motion-дизайнер", "моушн", "мошн",
    "анимация", "3d анимация", "2d анимация",
    "c4d", "cinema 4d", "blender", "ae",
    "эксплейнер", "аниматор", "motion graphics", "vfx",
    "композинг", "ролик", "shorts", "reels", "инфографика",
    "3d motion designer", "3d motion дизайнер",
    # маркеры заявки/задачи
    "бюджет", "сроки", "бриф", "тз", "оплата", "рейты",
    "стоимость", "контракт", "коммерческое предложение", "нужен", "требуется",
    "hiring", "hire", "need", "seeking",
]
NEGATIVE_WORDS = [
    "ищу работу", "ищу заказы", "возьму заказ", "готов выполнить",
    "выполню", "сделаю дешево", "портфолио", "резюме",
    "ищу подработку", "ищу вакансию", "ищу проект", "ищу стажировку",
    "готов работать", "ищу клиентов",
    "looking for job", "looking for work", "available for work",
    "hire me", "seeking gigs", "open to work", "need clients", "resume", "cv",
]
DEFAULT_KEYWORDS = KEYWORDS.copy()
DEFAULT_NEGATIVE_WORDS = NEGATIVE_WORDS.copy()

# OpenAI (если ключа нет — fallback выключен)
OPENAI_ENABLED = (
    os.getenv("OPENAI_ENABLED", "1").lower() in {"1", "true", "yes"}
    and OPENAI_AVAILABLE
    and bool(os.getenv("OPENAI_API_KEY", "").strip())
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motionbot")

# ================== Состояние/Аудит ==================
state_lock = asyncio.Lock()
STATE_DEFAULT: Dict[str, Any] = {
    "enabled": False,
    "channels": [],             # ссылки/юзернеймы каналов
    "reply_text": DEFAULT_REPLY,
    "owner_id": 0,
    "keywords": DEFAULT_KEYWORDS,
    "negative_words": DEFAULT_NEGATIVE_WORDS,
    "tap": False,               # присылать отчёты по каждому сообщению владельцу
    "media_files": [],          # до 3 файлов для прикрепления к отклику
}
AUDIT_FILE = "./data/audit.jsonl"  # JSONL-файл с решениями

def ensure_dirs():
    pathlib.Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(STATE_FILE) or ".").mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(AUDIT_FILE) or ".").mkdir(parents=True, exist_ok=True)

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
    return re.sub(r"\s+", " ", s).strip().lower()

def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        x = x.strip().lower()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _parse_terms(blob: str) -> List[str]:
    x = blob.replace(",", "\n").replace(";", "\n")
    return _dedup_keep_order([p for p in x.splitlines() if p.strip()])

KEYWORDS_NORM: List[str] = []
NEGATIVE_WORDS_NORM: List[str] = []

def refresh_globals_from_state():
    """Подтянуть KEYWORDS/NEGATIVE_WORDS из state.json в рантайм и нормализовать."""
    global KEYWORDS, NEGATIVE_WORDS, KEYWORDS_NORM, NEGATIVE_WORDS_NORM
    st = load_state()
    KEYWORDS = _dedup_keep_order(st.get("keywords", DEFAULT_KEYWORDS))
    NEGATIVE_WORDS = _dedup_keep_order(st.get("negative_words", DEFAULT_NEGATIVE_WORDS))
    # предрассчёт нормализованных списков для корректного сопоставления (ё→е, регистр)
    KEYWORDS_NORM = [norm(k) for k in KEYWORDS]
    NEGATIVE_WORDS_NORM = [norm(k) for k in NEGATIVE_WORDS]

# ---- Аудит файл ----
def _write_audit_sync(entry: Dict[str, Any]):
    ensure_dirs()
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

async def write_audit(entry: Dict[str, Any]):
    await asyncio.to_thread(_write_audit_sync, entry)

def _read_tail_sync(n: int) -> List[Dict[str, Any]]:
    if not os.path.exists(AUDIT_FILE):
        return []
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    out: List[Dict[str, Any]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

async def read_tail(n: int) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_read_tail_sync, n)

def _read_one_sync(key: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(AUDIT_FILE):
        return None
    found: Optional[Dict[str, Any]] = None
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                if obj.get("key") == key:
                    found = obj
            except Exception:
                continue
    return found

async def read_one(key: str) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(_read_one_sync, key)

# =========== user.session из base64 (опционально) ===========
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

# ================== Клиенты ==================
if not API_ID or not API_HASH:
    raise SystemExit("TELEGRAM_API_ID/TELEGRAM_API_HASH required in .env")
if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN required in .env")

ensure_dirs()
maybe_restore_user_session_from_b64()

user_session_path = str(pathlib.Path(SESSIONS_DIR) / USER_SESSION)
bot_session_path = str(pathlib.Path(SESSIONS_DIR) / BOT_SESSION)

user_client = TelegramClient(user_session_path, API_ID, API_HASH)  # личный аккаунт
bot_client = TelegramClient(bot_session_path, API_ID, API_HASH)    # бот по токену

# OpenAI init
if OPENAI_ENABLED:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        log.info(f"OpenAI ON ({OPENAI_MODEL})")
    except Exception as e:
        OPENAI_ENABLED = False
        log.warning(f"OpenAI init failed → disabled: {e}")

# ================== Фильтры ==================
REQ_VERBS = re.compile(
    r"(ищем|ищу|нужен|нужна|нужны|требуется|в\s*поиск|на\s*проект|задача|ваканси|hiring|hire|need|seeking)",
    re.I,
)

def primary_pass(text: str) -> bool:
    """Есть ключевое слово и глагол запроса (а не просто болтовня)."""
    t = norm(text)
    if not any(k in t for k in KEYWORDS_NORM):
        return False
    if not REQ_VERBS.search(t):
        return False
    return True

def negative_pass(text: str) -> bool:
    t = norm(text)
    return any(n in t for n in NEGATIVE_WORDS_NORM)

async def openai_gate(text: str) -> bool:
    """Строгая двоичная проверка через OpenAI. True — это заказ (клиент ищет исполнителя)."""
    if not OPENAI_ENABLED:
        return False
    try:
        system = (
            "You are a strict binary classifier. Answer only YES or NO.\n"
            "Decide if the message is a CLIENT REQUEST looking to hire a motion/animation specialist "
            "(2D/3D, motion graphics, video), as opposed to a person advertising themselves or looking for a job. "
            "Input language may be RU/KZ/EN."
        )
        user = f"Message:\n{text}"
        resp = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
            )
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return out.startswith("Y")
    except Exception as e:
        log.warning(f"OpenAI check failed: {e}")
        return False

# ============== Разрешённые каналы (храним peer_id как event.chat_id) ==============
ALLOW_IDS: set[int] = set()
RESOLVED_TITLES: Dict[int, str] = {}
RESOLVE_LOCK = asyncio.Lock()

async def ensure_join(link_or_at: str) -> Optional[int]:
    """
    Подписывает user_client на канал/группу.
    Возвращает peer_id (как event.chat_id) или None.
    """
    try:
        link = link_or_at.strip()
        if link.startswith("@"):
            uname = link[1:]
            ent = await user_client.get_entity(uname)
            try:
                await user_client(JoinChannelRequest(ent))
            except UserAlreadyParticipantError:
                pass
            ent = await user_client.get_entity(uname)
            return int(get_peer_id(ent))
        elif "t.me/+" in link or "joinchat/" in link:
            # приватные инвайты
            hash_ = link.split("t.me/")[-1]
            hash_ = hash_.split("+", 1)[-1] if "+" in hash_ else hash_.rsplit("/", 1)[-1]
            try:
                res = await user_client(ImportChatInviteRequest(hash_))
            except UserAlreadyParticipantError:
                res = None
            # попытаться получить peer_id из ответа/диалогов
            try:
                if res and getattr(res, "chats", None):
                    for ch in res.chats:
                        try:
                            return int(get_peer_id(ch))
                        except Exception:
                            continue
                # запасной вариант: после импорта получить последний диалог по ссылке нельзя,
                # поэтому просто резолвим текущие чаты позже
            except Exception:
                pass
            return None
        elif link.startswith("https://t.me/"):
            uname = link.split("https://t.me/")[-1].strip("/")
            ent = await user_client.get_entity(uname)
            try:
                await user_client(JoinChannelRequest(ent))
            except UserAlreadyParticipantError:
                pass
            ent = await user_client.get_entity(uname)
            return int(get_peer_id(ent))
        else:
            ent = await user_client.get_entity(link)
            try:
                await user_client(JoinChannelRequest(ent))
            except UserAlreadyParticipantError:
                pass
            return int(get_peer_id(ent))
    except Exception as e:
        log.warning(f"Join failed for {link_or_at}: {e}")
        return None

async def resolve_channels():
    """Из ссылок/юзернеймов делаем set peer_id (как event.chat_id)."""
    global ALLOW_IDS, RESOLVED_TITLES
    async with RESOLVE_LOCK:
        st = load_state()
        channels = st.get("channels") or []
        RESOLVED_TITLES = {}
        if not channels:
            ALLOW_IDS = set()
            log.warning("CHANNELS пуст — слушаем всё (только логируем chat_id/title).")
            return

        peer_ids: List[int] = []
        for link in channels:
            uname = link
            if link.startswith("https://t.me/"):
                uname = link.split("https://t.me/")[-1].strip("/")
            try:
                ent = await user_client.get_entity(uname if not link.startswith("@") else uname)
                ents = ent if isinstance(ent, (list, tuple)) else [ent]
                for e in ents:
                    pid = int(get_peer_id(e))  # может быть отрицательным (-100...)
                    title = getattr(e, "title", None) or getattr(e, "username", None) or str(pid)
                    peer_ids.append(pid)
                    RESOLVED_TITLES[pid] = title
                    log.info(f"Resolved: {link} → peer_id={pid} | {e.__class__.__name__} | {title}")
            except Exception as ex:
                log.error(f"Resolve failed: {link} → {ex}")

        ALLOW_IDS = set(peer_ids)
        log.info(f"Listening peer IDs: {sorted(ALLOW_IDS)}")

# ============== Мониторинг (user_client) ==============
@user_client.on(events.NewMessage())  # скобки обязательны
async def watcher(event):
    key = f"{event.chat_id}:{event.id}"
    try:
        st = load_state()
        if not st.get("enabled"):
            return
        if ALLOW_IDS and (event.chat_id not in ALLOW_IDS):
            return

        raw = event.raw_text or ""
        text = raw.strip()
        tnorm = norm(text)
        chat_title = getattr(getattr(event, "chat", None), "title", None) or RESOLVED_TITLES.get(int(event.chat_id))
        if VERBOSE:
            log.info(f"[{key}] New in '{chat_title or event.chat_id}': {preview(raw)}")

        # ==== Классификация ====
        prim = primary_pass(tnorm)
        neg = negative_pass(tnorm)
        used_openai = False
        ai_ok = False

        ok = False
        reason = "none"
        if prim and not neg:
            ok = True
            reason = "primary_pass"
        else:
            used_openai = True
            ai_ok = await openai_gate(text)
            ok = ai_ok
            reason = "openai_yes" if ai_ok else ("openai_no" if used_openai else "no_match")

        # поиск цели отправки
        m = re.search(r"@([A-Za-z0-9_]{4,32})", raw) or re.search(r"t\.me/([A-Za-z0-9_]{4,32})", raw, re.I)
        sender = await event.get_sender()
        if m:
            target = f"@{m.group(1)}"
        elif sender and getattr(sender, "username", None):
            target = f"@{sender.username}"
        else:
            target = getattr(sender, "id", None)

        final = "skip"
        send_error = None

        if ok and target:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            if VERBOSE:
                log.info(f"[{key}] Sleep {delay:.1f}s before send")
            await asyncio.sleep(delay)
            try:
                reply_text = str(st.get("reply_text") or DEFAULT_REPLY)
                media_files = list((st.get("media_files") or [])[:3])
                if media_files:
                    # Отправляем первый файл с подписью, остальные без
                    first, rest = media_files[0], media_files[1:]
                    await user_client.send_file(target, first, caption=reply_text)  # type: ignore[arg-type]
                    for mf in rest:
                        try:
                            await user_client.send_file(target, mf)  # type: ignore[arg-type]
                            await asyncio.sleep(0.5)
                        except Exception:
                            continue
                else:
                    await user_client.send_message(target, reply_text)  # type: ignore[arg-type]
                final = "sent"
                if st.get("owner_id"):
                    who = target if isinstance(target, str) else f"id:{target}"
                    src = chat_title or str(event.chat_id)
                    asyncio.create_task(
                        bot_client.send_message(st["owner_id"], f"✅ Отклик отправлен {who} (источник: {src})")
                    )
            except FloodWaitError as e:
                send_error = f"FloodWait {e.seconds}s"
            except (UserPrivacyRestrictedError, UserIsBlockedError):
                send_error = "privacy/blocked"
            except (UsernameInvalidError, PeerIdInvalidError, ChatWriteForbiddenError) as e:
                send_error = f"cannot write: {e}"
            except Exception as e:
                send_error = f"send failed: {e}"
        elif ok and not target:
            reason = "no_target"

        # ==== Аудит ====
        entry = {
            "key": key,
            "chat_id": int(event.chat_id),
            "chat_title": chat_title,
            "preview": preview(raw, 220),
            "primary": prim,
            "negative": neg,
            "used_openai": used_openai,
            "openai_ok": ai_ok,
            "decision": final,
            "reason": reason,
            "target": target if isinstance(target, str) else (int(target) if target else None),
            "send_error": send_error,
            "delay": f"{MIN_DELAY}-{MAX_DELAY}",
            "ts": getattr(event.message, "date", None).isoformat() if getattr(event, "message", None) else None,
        }
        await write_audit(entry)

        # ==== TAP режим: прислать отчёт владельцу ====
        if st.get("tap") and st.get("owner_id"):
            mark = "🟢" if final == "sent" else "🟡" if ok and final != "sent" else "⚪" if prim or used_openai else "⚫"
            txt = (
                f"{mark} [{key}] {chat_title or event.chat_id}\n"
                f"▶ {preview(raw, 300)}\n\n"
                f"primary: {'✅' if prim else '❌'} | negative: {'✅' if neg else '❌'} | "
                f"OpenAI: {'✅' if ai_ok else ('—' if not used_openai else '❌')}\n"
                f"decision: {final} | reason: {reason}\n"
                f"target: {entry['target']} | err: {send_error}"
            )
            try:
                await bot_client.send_message(st["owner_id"], txt)
            except Exception:
                pass

        if VERBOSE and final == "sent":
            log.info(f"[{key}] SENT → {target}")
        if VERBOSE and send_error:
            log.warning(f"[{key}] {send_error}")

    except Exception as e:
        log.exception(f"[{key}] handler error: {e}")

# ================== Команды бота (bot_client) ==================
def is_owner(st: Dict[str, Any], user_id: int) -> bool:
    owner = int(st.get("owner_id") or 0)
    return owner == 0 or owner == user_id  # 0 → первый /start назначит владельца

@bot_client.on(events.NewMessage(pattern=r"^/start"))
async def cmd_start(event):
    st = load_state()
    if st.get("owner_id") in (0, None):
        async with state_lock:
            st["owner_id"] = event.sender_id
            await save_state(st)
    txt = (
        "Привет! Я помогу искать заказы и автоматически отвечать.\n\n"
        "Основные команды:\n"
        "/on — включить мониторинг\n"
        "/off — выключить\n"
        "/add_channel <ссылка|@username> — добавить канал/группу\n"
        "/list_channels — показать список каналов\n"
        "/remove_channel <номер> — удалить канал из списка\n"
        "/set_reply <текст> — задать текст отклика\n"
        "/get_reply — показать текущий текст отклика\n"
        "/set_delay <min> <max> — задержка перед отправкой (в сек)\n"
        "/status — текущее состояние\n"
        "/tap_on — присылать краткий отчёт по каждому новому посту\n"
        "/tap_off — выключить отчёты"
    )
    await event.respond(txt)

@bot_client.on(events.NewMessage(pattern=r"^/ping$"))
async def cmd_ping(event):
    await event.respond("pong ✅")

@bot_client.on(events.NewMessage(pattern=r"^/whoami$"))
async def cmd_whoami(event):
    me = await user_client.get_me()
    uname = f"@{me.username}" if getattr(me, "username", None) else "(без username)"
    phone = getattr(me, "phone", None) or "—"
    await event.respond(f"User-client: {uname}\nID: {me.id}\nPhone: +{phone}")

@bot_client.on(events.NewMessage(pattern=r"^/on$"))
async def cmd_on(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["enabled"] = True
        await save_state(st)
    await event.respond("Мониторинг включён ✅")

@bot_client.on(events.NewMessage(pattern=r"^/off$"))
async def cmd_off(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["enabled"] = False
        await save_state(st)
    await event.respond("Мониторинг выключен ⏸️")

@bot_client.on(events.NewMessage(pattern=r"^/tap_on$"))
async def cmd_tap_on(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["tap"] = True
        await save_state(st)
    await event.respond("TAP режим включён — присылаю отчёты 👀")

@bot_client.on(events.NewMessage(pattern=r"^/tap_off$"))
async def cmd_tap_off(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["tap"] = False
        await save_state(st)
    await event.respond("TAP режим выключен.")

@bot_client.on(events.NewMessage(pattern=r"^/rescan$"))
async def cmd_rescan(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    await resolve_channels()
    await event.respond("Каналы перечитаны.")

@bot_client.on(events.NewMessage(pattern=r"^/ids$"))
async def cmd_ids(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return
    if not ALLOW_IDS:
        return await event.respond("ID не заданы (мониторинг всех чатов) или ещё не резолвились.")
    lines = [f"{pid}: {RESOLVED_TITLES.get(pid, '?')}" for pid in sorted(ALLOW_IDS)]
    await event.respond("Слушаем:\n" + "\n".join(lines))

@bot_client.on(events.NewMessage(pattern=r"^/join_all$"))
async def cmd_join_all(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return
    links = st.get("channels") or []
    ok = 0
    for l in links:
        cid = await ensure_join(l)
        if cid:
            ok += 1
        await asyncio.sleep(0.5)
    await resolve_channels()
    await event.respond(f"Подписка завершена. ОК: {ok}, всего: {len(links)}")

@bot_client.on(events.NewMessage(pattern=r"^/add_channel\s+(.+)$"))
async def cmd_add_channel(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    link = event.pattern_match.group(1).strip()
    async with state_lock:
        if link not in st["channels"]:
            st["channels"].append(link)
            await save_state(st)
    await event.respond(f"Добавил: {link}\nПодписываюсь…")
    await ensure_join(link)
    await resolve_channels()
    await event.respond("Готово. Проверь /ids")

@bot_client.on(events.NewMessage(pattern=r"^/list_channels$"))
async def cmd_list_channels(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    if not st["channels"]:
        return await event.respond("Список пуст.")
    body = "\n".join(f"{i+1}. {c}" for i, c in enumerate(st["channels"]))
    await event.respond(f"Каналы:\n{body}")

@bot_client.on(events.NewMessage(pattern=r"^/remove_channel\s+(\d+)$"))
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

@bot_client.on(events.NewMessage(pattern=r"^/set_reply\s+([\s\S]+)$"))
async def cmd_set_reply(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    txt = event.pattern_match.group(1)
    async with state_lock:
        st["reply_text"] = txt
        await save_state(st)
    await event.respond("Шаблон отклика обновлён ✅")

@bot_client.on(events.NewMessage(pattern=r"^/get_reply$"))
async def cmd_get_reply(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    await event.respond("Текущий шаблон:\n\n" + (st.get("reply_text") or DEFAULT_REPLY))

@bot_client.on(events.NewMessage(pattern=r"^/set_delay\s+(\d+)\s+(\d+)$"))
async def cmd_set_delay(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    a = int(event.pattern_match.group(1))
    b = int(event.pattern_match.group(2))
    global MIN_DELAY, MAX_DELAY
    MIN_DELAY = min(a, b)
    MAX_DELAY = max(a, b)
    await event.respond(f"Задержка обновлена: {MIN_DELAY}–{MAX_DELAY} сек")

@bot_client.on(events.NewMessage(pattern=r"^/status$"))
async def cmd_status(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")

    ch = "\n".join(f"- {c}" for c in st["channels"]) or "пусто"
    reply_preview = (st.get("reply_text") or DEFAULT_REPLY).strip().replace("\n", " ")
    if len(reply_preview) > 160:
        reply_preview = reply_preview[:160] + "…"

    await event.respond(
        "Состояние:\n"
        f"Мониторинг: {'включён ✅' if st['enabled'] else 'выключен ⏸️'}\n"
        f"Каналы:\n{ch}\n"
        f"Задержка: {MIN_DELAY}-{MAX_DELAY} c\n"
        f"Текст отклика (превью): {reply_preview}"
    )


# ===== Словари: list/add/del/reset =====
@bot_client.on(events.NewMessage(pattern=r"^/list_kw$"))
async def cmd_list_kw(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    kws = st.get("keywords", [])
    if not kws:
        return await event.respond("Ключевые слова: пусто")
    body = "\n".join(f"- {k}" for k in kws[:100])
    suffix = "" if len(kws) <= 100 else f"\n… и ещё {len(kws) - 100}"
    await event.respond(f"Ключевые слова ({len(kws)}):\n{body}{suffix}")

@bot_client.on(events.NewMessage(pattern=r"^/list_bad_kw$"))
async def cmd_list_bad_kw(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    bad = st.get("negative_words", [])
    if not bad:
        return await event.respond("Негативные слова: пусто")
    body = "\n".join(f"- {k}" for k in bad[:100])
    suffix = "" if len(bad) <= 100 else f"\n… и ещё {len(bad) - 100}"
    await event.respond(f"Негативные слова ({len(bad)}):\n{body}{suffix}")

@bot_client.on(events.NewMessage(pattern=r"^/add_kw\s+([\s\S]+)$"))
async def cmd_add_kw(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    terms = _parse_terms(event.pattern_match.group(1))
    if not terms:
        return await event.respond("Формат: /add_kw слово1;слово2;слово3")
    async with state_lock:
        cur = _dedup_keep_order(list(st.get("keywords", [])) + terms)
        st["keywords"] = cur
        await save_state(st)
    refresh_globals_from_state()
    await event.respond(f"Добавлено: {len(terms)}. Теперь ключевых: {len(st['keywords'])}")

@bot_client.on(events.NewMessage(pattern=r"^/add_bad_kw\s+([\s\S]+)$"))
async def cmd_add_bad_kw(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    terms = _parse_terms(event.pattern_match.group(1))
    if not terms:
        return await event.respond("Формат: /add_bad_kw слово1;слово2;слово3")
    async with state_lock:
        cur = _dedup_keep_order(list(st.get("negative_words", [])) + terms)
        st["negative_words"] = cur
        await save_state(st)
    refresh_globals_from_state()
    await event.respond(f"Добавлено: {len(terms)}. Теперь негативных: {len(st['negative_words'])}")

@bot_client.on(events.NewMessage(pattern=r"^/del_kw\s+(.+)$"))
async def cmd_del_kw(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    term = event.pattern_match.group(1).strip().lower()
    async with state_lock:
        arr = [k for k in st.get("keywords", []) if k.lower() != term]
        removed = len(st.get("keywords", [])) - len(arr)
        st["keywords"] = arr
        await save_state(st)
    refresh_globals_from_state()
    await event.respond(f"Удалено: {removed}")

@bot_client.on(events.NewMessage(pattern=r"^/del_bad_kw\s+(.+)$"))
async def cmd_del_bad_kw(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    term = event.pattern_match.group(1).strip().lower()
    async with state_lock:
        arr = [k for k in st.get("negative_words", []) if k.lower() != term]
        removed = len(st.get("negative_words", [])) - len(arr)
        st["negative_words"] = arr
        await save_state(st)
    refresh_globals_from_state()
    await event.respond(f"Удалено: {removed}")

@bot_client.on(events.NewMessage(pattern=r"^/reset_kw$"))
async def cmd_reset_kw(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["keywords"] = DEFAULT_KEYWORDS.copy()
        st["negative_words"] = DEFAULT_NEGATIVE_WORDS.copy()
        await save_state(st)
    refresh_globals_from_state()
    await event.respond("Словари сброшены к значениям по умолчанию ✅")

# ===== Диагностика аудита =====
@bot_client.on(events.NewMessage(pattern=r"^/tail(?:\s+(\d+))?$"))
async def cmd_tail(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return
    n = int(event.pattern_match.group(1) or 20)
    rows = await read_tail(max(1, min(n, 100)))
    if not rows:
        return await event.respond("Аудит пуст.")
    lines = []
    for r in rows:
        lines.append(
            f"[{r.get('key')}] {r.get('chat_title') or r.get('chat_id')} | "
            f"prim:{'1' if r.get('primary') else '0'} neg:{'1' if r.get('negative') else '0'} "
            f"ai:{'1' if r.get('openai_ok') else ('0' if r.get('used_openai') else '-') } "
            f"→ {r.get('decision')} ({r.get('reason')}) | tgt:{r.get('target')} | "
            f"{r.get('preview')}"
        )
    await event.respond("Последние записи:\n" + "\n".join(lines))

@bot_client.on(events.NewMessage(pattern=r"^/show\s+(-?\d+:\d+)$"))
async def cmd_show(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return
    key = event.pattern_match.group(1)
    row = await read_one(key)
    if not row:
        return await event.respond("Не нашёл запись в аудите.")
    txt = (
        f"[{row.get('key')}] {row.get('chat_title') or row.get('chat_id')}\n"
        f"primary: {row.get('primary')} | negative: {row.get('negative')} | "
        f"OpenAI used: {row.get('used_openai')} ok: {row.get('openai_ok')}\n"
        f"decision: {row.get('decision')} ({row.get('reason')}) | target: {row.get('target')}\n"
        f"send_error: {row.get('send_error')}\n\n"
        f"{row.get('preview')}"
    )
    await event.respond(txt)

# ===== Медиа-файлы для прикрепления к отклику =====
@bot_client.on(events.NewMessage(pattern=r"^/add_media\s+([\s\S]+)$"))
async def cmd_add_media(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    # допускаем до 3 файлов (локальные пути или http/https)
    items = _parse_terms(event.pattern_match.group(1))
    if not items:
        return await event.respond("Формат: /add_media url1;url2;path3")
    async with state_lock:
        cur = list(st.get("media_files", []))
        for it in items:
            if len(cur) >= 3:
                break
            if it not in cur:
                cur.append(it)
        st["media_files"] = cur[:3]
        await save_state(st)
    await event.respond(f"Файлы обновлены ({len(st['media_files'])}/3). Использую при отклике.")

@bot_client.on(events.NewMessage(pattern=r"^/list_media$"))
async def cmd_list_media(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    media = st.get("media_files", [])
    if not media:
        return await event.respond("Медиа не заданы.")
    body = "\n".join(f"- {m}" for m in media)
    await event.respond("Медиа (до 3):\n" + body)

@bot_client.on(events.NewMessage(pattern=r"^/clear_media$"))
async def cmd_clear_media(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return await event.respond("Только владелец может управлять.")
    async with state_lock:
        st["media_files"] = []
        await save_state(st)
    await event.respond("Медиа очищены ✅")

# ===== Экспорт аудита в Excel =====
@bot_client.on(events.NewMessage(pattern=r"^/export_xlsx$"))
async def cmd_export_xlsx(event):
    st = load_state()
    if not is_owner(st, event.sender_id):
        return
    try:
        import openpyxl  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
    except Exception:
        return await event.respond("openpyxl не установлен. Установите пакет и повторите.")

    rows = await read_tail(10000)  # экспортируем последние 10k записей
    if not rows:
        return await event.respond("Аудит пуст.")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "audit"
    headers = [
        "key", "chat_id", "chat_title", "preview", "primary", "negative",
        "used_openai", "openai_ok", "decision", "reason", "target", "send_error", "delay", "ts",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    # автоширина
    for idx, _ in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(idx)].width = 18

    ensure_dirs()
    out_path = os.path.join(os.path.dirname(AUDIT_FILE) or ".", "audit.xlsx")
    wb.save(out_path)
    try:
        await bot_client.send_file(event.chat_id, out_path)
    except Exception:
        return await event.respond(f"Сохранено: {out_path}")

# ================== Старт ==================
async def main():
    logging.getLogger("telethon").setLevel(logging.WARNING)

    log.info("=== CONFIG ===")
    log.info(f"Delay: {MIN_DELAY}-{MAX_DELAY}s | VERBOSE={VERBOSE}")
    log.info(f"Sessions dir: {SESSIONS_DIR} | USER={USER_SESSION} | BOT={BOT_SESSION}")
    log.info(f"State: {STATE_FILE}")
    log.info(f"OpenAI: {'ON' if OPENAI_ENABLED else 'OFF'}")

    # подтянуть словари из state.json (если ранее меняли командами)
    refresh_globals_from_state()

    # Первый запуск спросит номер/код/2FA в консоли
    await bot_client.start(bot_token=BOT_TOKEN)
    await user_client.start()

    await resolve_channels()
    log.info("Both clients started. Waiting for events...")

    await asyncio.gather(
        bot_client.run_until_disconnected(),
        user_client.run_until_disconnected(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
