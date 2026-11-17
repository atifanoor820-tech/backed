"""Configuration and storage for Dashi (ShowMyTrades trading agent)."""

import json
import re
import uuid
from pathlib import Path
from typing import Any

from agent.config import get_settings


def _config_path() -> Path:
    return get_settings().data_dir / "dashi_config.json"


def _default_config() -> dict[str, Any]:
    return {
        "active_agent": "abbas",  # "abbas" | "dashi"
        "profiles": [
            {
                "id": "profile_1",
                "name": "ShowMyTrades Profile 1",
                "email": "",
                "password": "",
                "profile_url": "https://showmytrades.com/",
                "notes": "Main profile",
            },
            {
                "id": "profile_2",
                "name": "ShowMyTrades Profile 2",
                "email": "",
                "password": "",
                "profile_url": "https://showmytrades.com/",
                "notes": "Secondary profile",
            },
            {
                "id": "profile_3",
                "name": "ShowMyTrades Profile 3",
                "email": "",
                "password": "",
                "profile_url": "https://showmytrades.com/",
                "notes": "Profile 3",
            },
            {
                "id": "profile_4",
                "name": "ShowMyTrades Profile 4",
                "email": "",
                "password": "",
                "profile_url": "https://showmytrades.com/",
                "notes": "Profile 4",
            },
            {
                "id": "profile_5",
                "name": "ShowMyTrades Profile 5",
                "email": "",
                "password": "",
                "profile_url": "https://showmytrades.com/",
                "notes": "Profile 5",
            },
        ],
        "trading_accounts": [
            {
                "id": "acc_1",
                "name": "Primary Trading Account",
                "account_number": "",
                "broker": "Exness",
                "server": "",
                "platform": "MT5",
                "login": "",
                "password": "",
                "notes": "",
            }
        ],
    }


def load_dashi_config() -> dict[str, Any]:
    path = _config_path()
    if not path.is_file():
        cfg = _default_config()
        save_dashi_config(cfg)
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        defaults = _default_config()
        for k, v in defaults.items():
            data.setdefault(k, v)
        if get_settings().dashi_only:
            data["active_agent"] = "dashi"
        return data
    except Exception:
        return _default_config()


def save_dashi_config(data: dict[str, Any]) -> dict[str, Any]:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return data


def is_dashi_active() -> bool:
    return load_dashi_config().get("active_agent") == "dashi"


def set_active_agent(agent_name: str) -> str:
    name = agent_name.strip().lower()
    if name not in ("abbas", "dashi"):
        raise ValueError(f"Unknown agent '{agent_name}'. Choose 'abbas' or 'dashi'.")
    if name == "abbas" and get_settings().dashi_only:
        raise ValueError("Abbas is turned off for now (ABBAS_DASHI_ONLY). Only Dashi is available.")
    cfg = load_dashi_config()
    cfg["active_agent"] = name
    save_dashi_config(cfg)
    return name


# --- Saved trading accounts (the "container") ----------------------------------------------------

ACCOUNT_FIELDS = ("name", "account_number", "broker", "server", "platform", "login", "password", "notes", "profile_id")
_NUM_RE = re.compile(r"(?<![\w.])(\d{5,12})(?![\w.])")
_LABEL_RE = {
    "password": r"(?:investor\s+)?(?:password|passwd|pass|pwd|pw|inv)",
    "server": r"server|srv",
    "broker": r"broker",
    "account_number": r"(?:account|acc|acct|login|id)(?:\s*(?:no|number|#))?",
}
_ANY_LABEL = "|".join(_LABEL_RE.values())
_SERVER_HINT = re.compile(r"(mt[45]|demo|live|real|server|-)", re.I)
# Words of the command around the credentials ("ye accounts container mein save karo"), never a password.
_COMMAND_WORD = re.compile(
    r"(?i)\d\)|mt[45]|accounts?|acc|save|store|karo|kro|krdo|kar|do|ye|yeh|ya|is|isko|ko|ka|ki|ke|mein|main|me|"
    r"container|settings?|vault|update|refresh|delete|remove|add|connect|dashi|please|pls|plz|and|aur|the|to|in|"
    r"on|of|my|mera|meray|mere|hai|hain|sab|sary|saare|sare|new|naya|dobara|again|now|abhi|lo|lena|hata|hatao|"
    r"rakho|rakh|yaad|daal|daalo|dalo|showmytrades|trading|here|these|this|is|are")


def _norm_number(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def find_account(cfg: dict[str, Any], number: str) -> dict[str, Any] | None:
    number = _norm_number(number)
    if not number:
        return None
    return next((a for a in cfg.get("trading_accounts", [])
                 if number in (_norm_number(a.get("account_number")), _norm_number(a.get("login")))), None)


def _broker_from_server(server: str) -> str:
    return re.split(r"[-\s]", server.strip(), maxsplit=1)[0] if server.strip() else ""


def parse_accounts_text(text: str) -> list[dict[str, str]]:
    """Read free text like '592235145 / Abc@123 / FxPro-MT5 Demo' or 'login: 5922 password: x server: y'
    (one account per line or per ';') into account dicts. Lines without an account number extend the last one."""
    accounts: list[dict[str, str]] = []
    for chunk in re.split(r"[\r\n;]+", text):
        chunk = chunk.strip(" \t-•*")
        if not chunk:
            continue
        found: dict[str, str] = {}
        rest = chunk
        for field, label in _LABEL_RE.items():
            value_re = rf"(.+?)(?=\s+[,|/]?\s*(?:{_ANY_LABEL})\b|\s+-\s+|\s*[,|]|$)" if field == "server" else r"(\S+)"
            m = re.search(rf"\b(?:{label})\b\s*[:=\-]?\s*{value_re}", rest, re.I)
            if m and m.group(1).strip(" ,|"):
                found[field] = m.group(1).strip(" ,|")
                rest = rest[:m.start()] + " | " + rest[m.end():]
        if "account_number" not in found and (m := _NUM_RE.search(rest)):
            found["account_number"] = m.group(1)
            rest = rest[:m.start()] + " | " + rest[m.end():]
        # Unlabelled leftovers: "5922 / Abc@123 / FxPro-MT5 Demo" or "5922 Abc@123 FxPro-MT5 Demo"
        words = [w for w in re.split(r"[\s|/,]+|\s+-\s+", rest)
                 if re.search(r"\w", w) and not _COMMAND_WORD.fullmatch(w.strip(":.!?"))]
        i = 0
        while i < len(words):
            w = words[i]
            if "server" not in found and _SERVER_HINT.search(w):
                server = [w]
                while i + 1 < len(words) and re.fullmatch(r"(?i)demo|live|real\d*|server\d*|\d{1,2}", words[i + 1]):
                    i += 1
                    server.append(words[i])
                found["server"] = " ".join(server)
            elif "password" not in found:
                found["password"] = w
            i += 1
        if found.get("account_number"):
            found["account_number"] = _norm_number(found["account_number"])
            accounts.append(found)
        elif accounts:
            for k, v in found.items():
                accounts[-1].setdefault(k, v)
    for acc in accounts:
        if acc.get("server") and not acc.get("broker"):
            acc["broker"] = _broker_from_server(acc["server"])
        if re.search(r"(?i)\bmt4\b", acc.get("server", "")):
            acc.setdefault("platform", "MT4")
    return accounts


def upsert_trading_accounts(incoming: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Save accounts by account number. Known numbers get only the non-empty fields given; returns (added, updated)."""
    cfg = load_dashi_config()
    accounts = cfg.setdefault("trading_accounts", [])
    added, updated = [], []
    for item in incoming:
        number = _norm_number(item.get("account_number") or item.get("login"))
        if not number:
            continue
        clean = {k: str(item[k]).strip() for k in ACCOUNT_FIELDS if item.get(k) not in (None, "")}
        clean["account_number"] = number
        if existing := find_account(cfg, number):
            changed = {k: v for k, v in clean.items() if existing.get(k) != v}
            if changed:
                existing.update(changed)
                updated.append(number)
        else:
            server = clean.get("server", "")
            accounts.append({
                "id": f"acc_{uuid.uuid4().hex[:8]}",
                "name": clean.get("name") or f"{clean.get('broker') or _broker_from_server(server) or 'Account'} {number}",
                "account_number": number,
                "broker": clean.get("broker") or _broker_from_server(server),
                "server": server,
                "platform": clean.get("platform", "MT5"),
                "login": clean.get("login", ""),
                "password": clean.get("password", ""),
                "notes": clean.get("notes", ""),
                "profile_id": clean.get("profile_id", ""),
            })
            added.append(number)
    if added or updated:
        save_dashi_config(cfg)
    return added, updated


def remember_account_profile(number: str, profile_id: str) -> None:
    """Note which ShowMyTrades profile holds the account, so the next update tries that profile first."""
    cfg = load_dashi_config()
    if (acc := find_account(cfg, number)) and profile_id and acc.get("profile_id") != profile_id:
        acc["profile_id"] = profile_id
        save_dashi_config(cfg)
