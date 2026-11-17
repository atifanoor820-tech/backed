"""Dashi agent tools: ShowMyTrades profile scanning, account deletion, and account addition."""

import re
import time
from typing import Any

from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.dashi_config import (find_account, load_dashi_config, parse_accounts_text, remember_account_profile,
                                set_active_agent, upsert_trading_accounts)
from agent.events import emit
from agent.state import Risk, ToolResult
from agent.tools.registry import Tool, ToolContext, register


def _step(task_id: str, group: str, key: str, label: str, state: str, detail: str = "") -> None:
    """Live checklist row for the chat: state is pending | running | done | failed | skipped."""
    emit(task_id, "progress", f"{group} · {label}: {state}" + (f" ({detail})" if detail else ""),
         {"progress": {"group": group, "key": key, "label": label, "state": state, "detail": detail}})


class NoArgs(BaseModel):
    pass


class SwitchAgentArgs(BaseModel):
    agent_name: str = Field(description="Agent name to switch to: 'abbas' or 'dashi'")


class DeleteAccountArgs(BaseModel):
    account_number: str = Field(default="", description="Trading account number/login to locate across profiles and delete")
    profile_id: str = Field(default="", description="Optional specific ShowMyTrades profile ID to target")


class AddAccountArgs(BaseModel):
    account_number: str = Field(default="", description="Trading account number/login to add")
    broker: str = Field(default="", description="Broker name (e.g. FxPro, Exness, IC Markets)")
    server: str = Field(default="", description="Broker trading server (e.g. FxPro-MT5 Demo)")
    password: str = Field(default="", description="Trading account investor password")
    platform: str = Field(default="MT5", description="Trading platform: MetaTrader 5 Direct Connection")
    profile_id: str = Field(default="", description="Optional specific ShowMyTrades profile ID to add to")


class UpdateAccountArgs(BaseModel):
    account_number: str = Field(default="", description="Trading account number/login to locate, delete and re-add")
    broker: str = Field(default="", description="Broker name (e.g. FxPro, Exness, IC Markets)")
    server: str = Field(default="", description="Broker trading server (e.g. FxPro-MT5 Demo)")
    login: str = Field(default="", description="Trading login ID (if different from account_number)")
    password: str = Field(default="", description="Trading account password (investor password)")
    platform: str = Field(default="MT5", description="Trading platform: MT4, MT5 Direct Connection")
    profile_id: str = Field(default="", description="Optional specific ShowMyTrades profile ID to target")


class CheckAccountsArgs(BaseModel):
    account_numbers: list[str] = Field(default_factory=list,
                                       description="Accounts to check; empty means every saved account")


class SavedAccountIn(BaseModel):
    account_number: str = Field(description="Trading account number / MT login")
    password: str = Field(default="", description="Investor password")
    server: str = Field(default="", description="Broker server, e.g. FxPro-MT5 Demo")
    broker: str = Field(default="", description="Broker name; derived from the server when empty")
    platform: str = Field(default="MT5", description="MT5 or MT4")
    name: str = Field(default="", description="Optional label")


class SaveAccountsArgs(BaseModel):
    accounts: list[SavedAccountIn] = Field(default_factory=list, description="Accounts the user gave, one item each")
    text: str = Field(default="", description="The user's raw message; parsed when accounts is empty")


class SaveProfileArgs(BaseModel):
    profile_id: str = Field(description="Profile identifier, e.g. profile_1")
    name: str = Field(description="Profile display name")
    email: str = Field(default="", description="ShowMyTrades login email")
    password: str = Field(default="", description="ShowMyTrades login password")
    profile_url: str = Field(default="https://showmytrades.com/", description="Profile URL")
    notes: str = Field(default="", description="Optional notes")


def _switch_agent(args: SwitchAgentArgs, ctx: ToolContext) -> ToolResult:
    try:
        name = set_active_agent(args.agent_name)
        msg = f"Switched active agent to '{name.title()}'"
        emit(ctx.task_id, "tool_completed", msg)
        return ToolResult(ok=True, summary=msg, data={"active_agent": name})
    except ValueError as e:
        return ToolResult(ok=False, summary=str(e))


def _list_dashi(args: NoArgs, ctx: ToolContext) -> ToolResult:
    cfg = load_dashi_config()
    profiles = cfg.get("profiles", [])
    accounts = cfg.get("trading_accounts", [])
    active = cfg.get("active_agent", "abbas")
    summary = (
        f"Active Agent: {active.title()} | Configured ShowMyTrades Profiles: {len(profiles)} | "
        f"Trading Accounts: {len(accounts)}"
    )
    return ToolResult(
        ok=True,
        summary=summary,
        data={"active_agent": active,
              "profiles": [{k: v for k, v in p.items() if k != "password"} for p in profiles],
              "trading_accounts": [{k: v for k, v in a.items() if k != "password"} for a in accounts]},
        evidence=[p.get("name", "Unnamed") for p in profiles]
        + [f"{a.get('account_number')} · {a.get('server') or a.get('broker') or '?'}" for a in accounts],
    )


def _resolve_account_details(raw_acc: str, broker: str, server: str, login: str, password: str, platform: str, stored_accounts: list[dict]):
    """Fills in missing account details from dashi_config.json stored accounts."""
    target_acc = raw_acc.strip()
    target_broker = broker.strip()
    target_server = server.strip()
    target_login = login.strip() or target_acc
    target_password = password.strip()
    target_platform = platform.strip() or "MT5"

    matched_acc = find_account({"trading_accounts": stored_accounts}, target_acc) if target_acc else None
    if not target_acc and len(stored_accounts) == 1:
        matched_acc = stored_accounts[0]  # "update account" with a single saved account is unambiguous

    if matched_acc:
        target_acc = target_acc or matched_acc.get("account_number") or matched_acc.get("login") or ""
        target_broker = target_broker or matched_acc.get("broker", "")
        target_server = target_server or matched_acc.get("server", "")
        target_login = target_login or matched_acc.get("login", "") or target_acc
        target_password = target_password or matched_acc.get("password", "")
        target_platform = target_platform or matched_acc.get("platform", "MT5")

    return target_acc, target_broker, target_server, target_login, target_password, target_platform


def _save_accounts(args: SaveAccountsArgs, ctx: ToolContext) -> ToolResult:
    """Store the accounts the user typed in chat, so later 'update 592235145' needs only the number."""
    incoming = [a.model_dump() for a in args.accounts if a.account_number.strip()] or parse_accounts_text(args.text)
    if not incoming:
        return ToolResult(ok=False, summary="I found no account numbers. Send them like: 592235145 / password / FxPro-MT5 Demo "
                                            "(one account per line).")
    added, updated = upsert_trading_accounts(incoming)
    cfg = load_dashi_config()
    missing = [a["account_number"] for a in cfg.get("trading_accounts", [])
               if a["account_number"] in {*added, *updated} and not (a.get("password") and a.get("server"))]
    parts = []
    if added:
        parts.append(f"saved {len(added)} new ({', '.join(added)})")
    if updated:
        parts.append(f"updated {len(updated)} ({', '.join(updated)})")
    summary = ("[OK] Dashi " + " and ".join(parts) if parts else "These accounts were already saved with the same details") \
        + f". {len(cfg.get('trading_accounts', []))} accounts in Settings → Dashi."
    if missing:
        summary += f" Missing password or server for: {', '.join(missing)}."
    emit(ctx.task_id, "tool_completed", summary)
    return ToolResult(ok=True, summary=summary, data={"added": added, "updated": updated, "incomplete": missing},
                      evidence=[f"{a['account_number']} · {a.get('server') or '?'}" for a in incoming])


def _remember_given(number: str, broker: str, server: str, password: str, platform: str) -> None:
    """Credentials typed with an add/update command go into the saved accounts too."""
    if number and (password or server):
        upsert_trading_accounts([{"account_number": number, "broker": broker, "server": server,
                                  "password": password, "platform": platform}])


def _profiles_known_first(profiles: list[dict], number: str, stored_accounts: list[dict]) -> list[dict]:
    acc = find_account({"trading_accounts": stored_accounts}, number) or {}
    known = acc.get("profile_id")
    return sorted(profiles, key=lambda p: p.get("id") != known) if known else profiles


def _login_profile(page, prof: dict, task_id: str) -> None:
    prof_name = prof.get("name", "Profile")
    prof_email = prof.get("email", "").strip()
    prof_pass = prof.get("password", "").strip()

    emit(task_id, "tool_started", f"Dashi opening ShowMyTrades profile: {prof_name}")
    page.goto("https://showmytrades.com/login", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(1500)

    # Dismiss cookie banner if present
    cookie_btn = page.locator('button:has-text("Accept"), button:has-text("Opt out")').first
    if cookie_btn.count() > 0 and cookie_btn.is_visible():
        try:
            cookie_btn.click()
        except Exception:
            pass

    if prof_email and prof_pass:
        email_inp = page.locator('input[name="email"], input[type="email"], input[placeholder*="email" i]').first
        if email_inp.count() > 0:
            email_inp.fill(prof_email)
        pass_inp = page.locator('input[name="password"], input[type="password"]').first
        if pass_inp.count() > 0:
            pass_inp.fill(prof_pass)
        sub_btn = page.locator('button[type="submit"], button:has-text("Sign in"), button:has-text("Continue")').first
        if sub_btn.count() > 0:
            sub_btn.click()
            page.wait_for_timeout(3500)


def _perform_delete(page, target_acc_num: str, shots, task_id: str, prof_name: str, idx: int) -> bool:
    """Locates account on accounts page, clicks 3 dots, clicks Delete account, confirms acknowledgment, permanently deletes."""
    if "dashboard/accounts" not in page.url:
        page.goto("https://showmytrades.com/dashboard/accounts", wait_until="domcontentloaded", timeout=25000)
        page.wait_for_timeout(2000)

    # Dismiss cookie banner
    cookie_btn = page.locator('button:has-text("Accept"), button:has-text("Opt out")').first
    if cookie_btn.count() > 0 and cookie_btn.is_visible():
        try:
            cookie_btn.click()
        except Exception:
            pass

    # Search for the account
    emit(task_id, "tool_started", f"Searching for account '{target_acc_num}' in {prof_name}...")
    search_box = page.locator('input[type="search"], input[placeholder*="search" i], input[placeholder*="filter" i]').first
    if search_box.count() > 0:
        search_box.fill(target_acc_num)
        page.wait_for_timeout(1500)

    target_row = page.locator(f'tr:has-text("{target_acc_num}")').first
    if target_row.count() == 0:
        return False

    # Snapshot before delete
    snap1 = shots / f"{task_id}_found_{idx}.png"
    page.screenshot(path=str(snap1))
    emit(task_id, "screenshot_captured", snap1.name, {"path": snap1.name})

    # Step: Click 3 dots
    emit(task_id, "tool_started", f"Opening actions menu (three dots) for '{target_acc_num}'...")
    trigger = target_row.locator('button[aria-haspopup="menu"], button[id*="dropdown"]').first
    if trigger.count() == 0:
        trigger = target_row.locator("button").last
    trigger.click()
    page.wait_for_timeout(800)

    # Step: Click 'Delete account' from dropdown menu
    emit(task_id, "tool_started", f"Clicking 'Delete account' for '{target_acc_num}'...")
    del_item = page.locator('button[role="menuitem"]:has-text("Delete account"), button:has-text("Delete account"), div:has-text("Delete account")').last
    if del_item.count() > 0:
        del_item.click()
        page.wait_for_timeout(1000)

    # Step: Check acknowledgment checkbox 'I understand this cannot be undone.'
    ack_checkbox = page.locator('button[data-testid="list-delete-ack"], [data-testid="list-delete-ack"], button:has-text("I understand")').first
    if ack_checkbox.count() > 0:
        ack_checkbox.click()
        page.wait_for_timeout(600)

    # Step: Click 'Delete account permanently'
    confirm_del_btn = page.locator('button[data-testid="list-delete-confirm"], button:has-text("Delete account permanently"), div[role="dialog"] button:has-text("Delete")').last
    if confirm_del_btn.count() > 0:
        confirm_del_btn.click()
        page.wait_for_timeout(3500)

    # Snapshot after deletion
    snap_del = shots / f"{task_id}_deleted_{idx}.png"
    page.screenshot(path=str(snap_del))
    emit(task_id, "screenshot_captured", snap_del.name, {"path": snap_del.name})
    return True


def _perform_add(page, target_acc_num: str, target_server: str, target_broker: str, target_password: str, shots, task_id: str, prof_name: str, idx: int) -> bool:
    """Navigates to accounts-new, selects MetaTrader 5 Direct Connection, fills Account Name, searches Broker Server, fills Account Number & Password, clicks Submit."""
    _step(task_id, target_acc_num, "add", "Add on ShowMyTrades", "running", f"filling the form in {prof_name}")
    emit(task_id, "tool_started", f"Opening Add Account (+ New account) on {prof_name}...")
    new_acc_btn = page.locator('button:has-text("New account"), a:has-text("New account"), button:has-text("+ New account"), a[href*="accounts-new"]').first
    if new_acc_btn.count() > 0:
        new_acc_btn.click()
        page.wait_for_timeout(2000)
    else:
        page.goto("https://showmytrades.com/dashboard/accounts-new", wait_until="domcontentloaded")
        page.wait_for_timeout(2000)

    # Dismiss cookie banner
    cookie_btn = page.locator('button:has-text("Accept"), button:has-text("Opt out")').first
    if cookie_btn.count() > 0 and cookie_btn.is_visible():
        try:
            cookie_btn.click()
        except Exception:
            pass

    # 1. Platform: Always MetaTrader 5 Direct Connection
    select_el = page.locator("select").first
    if select_el.count() > 0:
        try:
            select_el.select_option("mt5-direct")
            page.wait_for_timeout(1000)
        except Exception:
            pass

    # 2. Account Name: any friendly name
    acc_name_inp = page.locator('input[name="accountName"], input[placeholder*="Main Trading Account" i]').first
    if acc_name_inp.count() > 0:
        acc_name_inp.fill(f"Account {target_acc_num}")

    # 3. Broker Server: search and click match in dropdown
    server_query = target_server or target_broker or "FxPro"
    srv_btn = page.locator('button:has-text("Type to search server"), button[aria-haspopup="listbox"]').first
    if srv_btn.count() > 0:
        srv_btn.click()
        page.wait_for_timeout(600)
        srv_search = page.locator('input[role="combobox"]').last
        if srv_search.count() > 0:
            srv_search.fill(server_query)
            page.wait_for_timeout(1500)
            opts = page.locator('[role="option"], [data-reka-combobox-item]').all()
            if opts:
                opts[0].click()
                page.wait_for_timeout(800)

    # 4. Account Number
    acc_num_inp = page.locator('input[name="accountNumber"], input[placeholder*="12345678" i]').first
    if acc_num_inp.count() > 0:
        acc_num_inp.fill(target_acc_num)

    # 5. Investor Password
    pass_inp = page.locator('input[name="investorPassword"], input[type="password"]').last
    if pass_inp.count() > 0 and target_password:
        pass_inp.fill(target_password)

    # 6. Submit
    emit(task_id, "tool_started", f"Submitting account credentials for '{target_acc_num}'...")
    submit_btn = page.locator('button:has-text("Submit"), button[type="submit"]').last
    if submit_btn.count() > 0:
        submit_btn.click()
        page.wait_for_timeout(5000)

    # Verify on accounts page
    page.goto("https://showmytrades.com/dashboard/accounts", wait_until="domcontentloaded")
    page.wait_for_timeout(2000)

    # Capture snapshot after submitting and verifying on accounts list
    snap2 = shots / f"{task_id}_added_{idx}.png"
    page.screenshot(path=str(snap2))
    emit(task_id, "screenshot_captured", snap2.name, {"path": snap2.name})

    _step(task_id, target_acc_num, "add", "Add on ShowMyTrades", "done", prof_name)

    # A re-added account starts out Private on ShowMyTrades; the user wants it Published.
    _step(task_id, target_acc_num, "publish", "Set Published", "running", "opening account settings")
    try:
        ok = _make_published(page, target_acc_num, shots, task_id, idx)
    except Exception as e:  # the account is added either way; report the visibility step separately
        emit(task_id, "tool_failed", f"Could not set '{target_acc_num}' to Published: {str(e)[:120]}")
        ok = False
    _step(task_id, target_acc_num, "publish", "Set Published", "done" if ok else "failed",
          "" if ok else "change Visibility in its Settings")
    return ok


def _open_account_settings(page, target_acc_num: str, task_id: str) -> bool:
    """Opens the 'Account Settings' dialog for one account from the accounts list."""
    marker = page.get_by_text("Account Settings", exact=True).first

    def opened() -> bool:
        try:
            marker.wait_for(state="visible", timeout=5000)
            return True
        except Exception:
            return False

    search_box = page.locator('input[type="search"], input[placeholder*="search" i], input[placeholder*="filter" i]').first
    # A freshly added account can take a while to show up in the list: reload and search again until it does.
    row = page.locator(f'tr:has-text("{target_acc_num}")').first
    for attempt in range(12):
        search_box = page.locator('input[type="search"], input[placeholder*="search" i], input[placeholder*="filter" i]').first
        if search_box.count() > 0:
            search_box.fill(target_acc_num)
            page.wait_for_timeout(1500)
        if row.count() > 0:
            break
        page.wait_for_timeout(5000)
        page.goto("https://showmytrades.com/dashboard/accounts", wait_until="domcontentloaded", timeout=25000)
        page.wait_for_timeout(2500)
    if row.count() == 0:
        return False

    # The ⋮ menu has only "Copy account ID", "View account details" and "Delete account";
    # Settings lives on the account's own page.
    trigger = row.locator('button[aria-haspopup="menu"], button[id*="dropdown"]').first
    if trigger.count() == 0:
        trigger = row.locator("button").last
    trigger.click()
    page.wait_for_timeout(800)
    details = page.get_by_role("menuitem", name=re.compile(r"view account details", re.I)).first
    if details.count() == 0:
        details = page.locator('[role="menu"] :text-matches("account details", "i")').first
    if details.count() > 0:
        details.click()
    else:
        page.keyboard.press("Escape")
        link = row.locator("a").first
        if link.count() == 0:
            return False
        link.click()
    page.wait_for_timeout(3000)

    # A just-added account shows "Unable to load account" until ShowMyTrades has synced it: press Retry and wait.
    for _ in range(18):
        if page.get_by_text("Unable to load account").count() == 0:
            break
        emit(task_id, "tool_started", f"Account '{target_acc_num}' is still syncing on ShowMyTrades, retrying...")
        _step(task_id, target_acc_num, "publish", "Set Published", "running", "waiting for ShowMyTrades to sync")
        retry = page.locator('button:has-text("Retry")').first
        if retry.count() > 0:
            retry.click()
        else:
            page.reload(wait_until="domcontentloaded")
        page.wait_for_timeout(5000)
    page.wait_for_timeout(1500)

    settings_btn = page.locator(
        'button:has-text("Settings"), a:has-text("Settings"), button[aria-label*="setting" i], '
        'button[title*="setting" i], button:has(svg.lucide-settings), button:has(svg[class*="settings"]), '
        'button:has(svg[class*="cog"])').first
    if settings_btn.count() > 0:
        settings_btn.click()
        if opened():
            return True

    # The Private / Published pill in the page header may open the same settings.
    pill = page.locator('header button:has-text("Private"), button:has-text("Private"), '
                        'button:has-text("Unlisted")').first
    if pill.count() > 0:
        pill.click()
        if opened():
            return True

    labels = [t.strip() for t in page.locator("main button, header button").all_inner_texts() if t.strip()]
    emit(task_id, "tool_failed", f"No Settings button on the account page. Buttons seen: {labels[:15]}")
    return False


def _make_published(page, target_acc_num: str, shots, task_id: str, idx: int) -> bool:
    """Account Settings -> General -> Visibility = 'Published - Visible to others' -> Update."""
    emit(task_id, "tool_started", f"Opening Settings for '{target_acc_num}' to make it Published...")
    if not _open_account_settings(page, target_acc_num, task_id):
        snap = shots / f"{task_id}_settings_missing_{idx}.png"
        page.screenshot(path=str(snap))
        emit(task_id, "screenshot_captured", snap.name, {"path": snap.name})
        emit(task_id, "tool_failed", f"Could not open Settings for '{target_acc_num}'.")
        return False
    dialog = page.locator('[role="dialog"]:has-text("Account Settings")').first
    if dialog.count() == 0:
        dialog = page.locator("body")

    general = dialog.locator('button:has-text("General"), [role="tab"]:has-text("General")').first
    if general.count() > 0:
        general.click()
        page.wait_for_timeout(500)

    emit(task_id, "tool_started", f"Setting visibility of '{target_acc_num}' to Published...")
    try:  # the dialog's General tab renders a moment after it opens
        page.get_by_text("Visibility", exact=True).first.wait_for(state="visible", timeout=8000)
    except Exception:
        pass
    page.wait_for_timeout(800)

    picked = False
    # Native <select>: pick the option whose text says Published.
    for sel in page.locator("select").all():
        try:
            value = sel.evaluate("s => [...s.options].find(o => /published/i.test(o.textContent))?.value ?? null")
        except Exception:
            continue
        if value is not None:
            sel.select_option(value)
            picked = True
            break
    if not picked:  # a custom dropdown placed under the "Visibility" label
        box = page.locator('[role="combobox"]:has-text("Private"), [role="combobox"]:has-text("Unlisted"), '
                           'button:has-text("Private - Only you"), button:has-text("Unlisted - Only")').first
        if box.count() > 0:
            box.click()
            page.wait_for_timeout(700)
            opt = page.locator('[role="option"]:has-text("Published"), li:has-text("Published"), '
                               'div[role="menuitem"]:has-text("Published")').first
            if opt.count() > 0:
                opt.click()
                picked = True
        elif page.locator(':text("Published - Visible")').count() > 0:
            picked = True  # already Published
    page.wait_for_timeout(600)
    if not picked:
        snap = shots / f"{task_id}_visibility_missing_{idx}.png"
        page.screenshot(path=str(snap))
        emit(task_id, "screenshot_captured", snap.name, {"path": snap.name})
        boxes = page.locator('select, [role="combobox"]').all_inner_texts()
        emit(task_id, "tool_failed", f"Could not find the Visibility setting for '{target_acc_num}'. "
                                     f"Dropdowns seen: {[b.strip()[:60] for b in boxes][:8]}")
        return False

    save = dialog.locator('button:has-text("Update"), button:has-text("Save")').last
    if save.count() == 0:
        save = page.locator('button:has-text("Update"), button:has-text("Save changes"), button:has-text("Save")').last
    if save.count() == 0:
        emit(task_id, "tool_failed", "Could not find the Update button in Account Settings.")
        return False
    save.scroll_into_view_if_needed()
    save.click()
    page.wait_for_timeout(2500)

    snap = shots / f"{task_id}_published_{idx}.png"
    page.screenshot(path=str(snap))
    emit(task_id, "screenshot_captured", snap.name, {"path": snap.name})
    emit(task_id, "tool_completed", f"Account '{target_acc_num}' set to Published.")
    return True


def _delete_trading_account(args: DeleteAccountArgs, ctx: ToolContext) -> ToolResult:
    """Action: DELETE ONLY. Scans ShowMyTrades profiles, finds account, deletes it, does NOT add."""
    cfg = load_dashi_config()
    profiles = cfg.get("profiles", [])
    stored_accounts = cfg.get("trading_accounts", [])

    target_acc, _, _, _, _, _ = _resolve_account_details(
        args.account_number, "", "", "", "", "", stored_accounts
    )
    if not target_acc:
        return ToolResult(ok=False, summary="Please specify which account number to delete (e.g. 'delete account 592235145').")

    valid_profiles = [p for p in profiles if p.get("email") or p.get("profile_url")]
    if not valid_profiles:
        return ToolResult(ok=False, summary="No ShowMyTrades profiles configured yet! Please add profile credentials in Settings → Dashi.")

    from playwright.sync_api import sync_playwright
    from agent.desktop_env import ensure_default_desktop, bring_to_front
    from agent.tools.desktop import visible_windows

    ensure_default_desktop()
    shots = get_settings().data_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    found_profile = None
    inspected_profiles = []
    profiles = _profiles_known_first(profiles, target_acc, stored_accounts)

    try:
        with sync_playwright() as p:
            headless = get_settings().browser_headless
            try:
                browser = p.chromium.launch(headless=headless, channel="chrome")
            except Exception:
                browser = p.chromium.launch(headless=headless, channel=get_settings().browser_channel or "chrome")

            page = browser.new_page(viewport={"width": 1366, "height": 850})
            if not headless:
                time.sleep(0.5)
                wins = visible_windows()
                chrome_wins = [w for w in wins if "chrome" in w["title"].lower()]
                if chrome_wins:
                    bring_to_front(chrome_wins[0]["handle"])

            for idx, prof in enumerate(profiles):
                prof_name = prof.get("name", f"Profile {idx + 1}")
                inspected_profiles.append(prof_name)
                _step(ctx.task_id, target_acc, "find", "Find account", "running", f"looking in {prof_name}")
                _login_profile(page, prof, ctx.task_id)

                deleted = _perform_delete(page, target_acc, shots, ctx.task_id, prof_name, idx)
                if deleted:
                    _step(ctx.task_id, target_acc, "find", "Find account", "done", prof_name)
                    _step(ctx.task_id, target_acc, "delete", "Delete old connection", "done")
                    found_profile = prof_name
                    remember_account_profile(target_acc, prof.get("id", ""))
                    break
                else:
                    # Account not in this profile, clear cookies and proceed
                    page.context.clear_cookies()
                    try:
                        page.evaluate("() => { try { localStorage.clear(); sessionStorage.clear(); } catch(e){} }")
                    except Exception:
                        pass

            browser.close()
    except Exception as e:
        return ToolResult(ok=False, summary=f"Error deleting account: {str(e)[:150]}")

    if found_profile:
        summary = f"[OK] Dashi found account '{target_acc}' in '{found_profile}' and successfully deleted it."
    else:
        summary = f"Account '{target_acc}' was not found in any of the {len(inspected_profiles)} scanned profiles."

    return ToolResult(ok=True, summary=summary, data={"target_account": target_acc, "found_profile": found_profile, "scanned": inspected_profiles})


def _add_trading_account(args: AddAccountArgs, ctx: ToolContext) -> ToolResult:
    """Action: ADD ONLY. Navigates to accounts-new, fills credentials, clicks submit. Does NOT delete."""
    cfg = load_dashi_config()
    profiles = cfg.get("profiles", [])
    stored_accounts = cfg.get("trading_accounts", [])

    target_acc, target_broker, target_server, _, target_password, target_platform = _resolve_account_details(
        args.account_number, args.broker, args.server, "", args.password, args.platform, stored_accounts
    )
    if not target_acc:
        return ToolResult(ok=False, summary="Please specify the account number to add (e.g. 'add account 592235145').")
    _remember_given(target_acc, args.broker, args.server, args.password, args.platform)
    if not target_password:
        return ToolResult(ok=False, summary=f"No saved password for account {target_acc}. Send it once, e.g. "
                                            f"'{target_acc} / password / server', and I'll keep it in Settings → Dashi.")

    valid_profiles = [p for p in profiles if p.get("email") or p.get("profile_url")]
    if not valid_profiles:
        return ToolResult(ok=False, summary="No ShowMyTrades profiles configured yet! Please add profile credentials in Settings → Dashi.")

    from playwright.sync_api import sync_playwright
    from agent.desktop_env import ensure_default_desktop, bring_to_front
    from agent.tools.desktop import visible_windows

    ensure_default_desktop()
    shots = get_settings().data_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    # Use specified profile or first profile
    target_prof = valid_profiles[0]
    if args.profile_id:
        match_prof = next((p for p in valid_profiles if p.get("id") == args.profile_id), None)
        if match_prof:
            target_prof = match_prof

    prof_name = target_prof.get("name", "Profile 1")

    try:
        with sync_playwright() as p:
            headless = get_settings().browser_headless
            try:
                browser = p.chromium.launch(headless=headless, channel="chrome")
            except Exception:
                browser = p.chromium.launch(headless=headless, channel=get_settings().browser_channel or "chrome")

            page = browser.new_page(viewport={"width": 1366, "height": 850})
            if not headless:
                time.sleep(0.5)
                wins = visible_windows()
                chrome_wins = [w for w in wins if "chrome" in w["title"].lower()]
                if chrome_wins:
                    bring_to_front(chrome_wins[0]["handle"])

            _login_profile(page, target_prof, ctx.task_id)
            published = _perform_add(page, target_acc, target_server, target_broker, target_password, shots, ctx.task_id, prof_name, 0)
            browser.close()
        remember_account_profile(target_acc, target_prof.get("id", ""))
    except Exception as e:
        return ToolResult(ok=False, summary=f"Error adding account: {str(e)[:150]}")

    summary = f"[OK] Dashi successfully added account '{target_acc}' (MetaTrader 5 Direct Connection / {target_server or target_broker}) to '{prof_name}'." + (
        "\n[OK] Visibility set to Published." if published
        else "\n[!] Added, but could not set it to Published. Change Visibility in the account's Settings.")
    return ToolResult(ok=True, summary=summary, data={"target_account": target_acc, "profile": prof_name, "server": target_server})


def _update_trading_account(args: UpdateAccountArgs, ctx: ToolContext) -> ToolResult:
    """Action: UPDATE (DELETE OLD + RE-ADD FRESH). Scans profiles, deletes existing account, then re-adds with fresh credentials."""
    cfg = load_dashi_config()
    profiles = cfg.get("profiles", [])
    stored_accounts = cfg.get("trading_accounts", [])

    target_acc, target_broker, target_server, target_login, target_password, target_platform = _resolve_account_details(
        args.account_number, args.broker, args.server, args.login, args.password, args.platform, stored_accounts
    )
    if not target_acc:
        return ToolResult(ok=False, summary="Please specify the trading account number to update (e.g. 'update account 592235145').")
    _remember_given(target_acc, args.broker, args.server, args.password, args.platform)
    if not (target_password and (target_server or target_broker)):
        # Checked before deleting: an update that can't re-add would just lose the account.
        return ToolResult(ok=False, summary=f"Account {target_acc} has no saved password/server, so I didn't delete anything. "
                                            f"Send '{target_acc} / password / server' once and I'll save it, then say update again.")

    valid_profiles = [p for p in profiles if p.get("email") or p.get("profile_url")]
    if not valid_profiles:
        return ToolResult(ok=False, summary="No ShowMyTrades profiles configured yet! Please add profile credentials in Settings → Dashi.")

    from playwright.sync_api import sync_playwright
    from agent.desktop_env import ensure_default_desktop, bring_to_front
    from agent.tools.desktop import visible_windows

    ensure_default_desktop()
    shots = get_settings().data_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    found_profile = None
    inspected_profiles = []
    actions_taken = []
    profiles = _profiles_known_first(profiles, target_acc, stored_accounts)
    for key, label in (("find", "Find account"), ("delete", "Delete old connection"),
                       ("add", "Add on ShowMyTrades"), ("publish", "Set Published")):
        _step(ctx.task_id, target_acc, key, label, "pending")

    try:
        with sync_playwright() as p:
            headless = get_settings().browser_headless
            try:
                browser = p.chromium.launch(headless=headless, channel="chrome")
            except Exception:
                browser = p.chromium.launch(headless=headless, channel=get_settings().browser_channel or "chrome")

            page = browser.new_page(viewport={"width": 1366, "height": 850})
            if not headless:
                time.sleep(0.5)
                wins = visible_windows()
                chrome_wins = [w for w in wins if "chrome" in w["title"].lower()]
                if chrome_wins:
                    bring_to_front(chrome_wins[0]["handle"])

            for idx, prof in enumerate(profiles):
                prof_name = prof.get("name", f"Profile {idx + 1}")
                inspected_profiles.append(prof_name)
                _login_profile(page, prof, ctx.task_id)

                deleted = _perform_delete(page, target_acc, shots, ctx.task_id, prof_name, idx)
                if deleted:
                    found_profile = prof_name
                    remember_account_profile(target_acc, prof.get("id", ""))
                    actions_taken.append(f"Deleted account '{target_acc}' from {prof_name}")

                    # Immediately re-add on this profile
                    published = _perform_add(page, target_acc, target_server, target_broker, target_password, shots, ctx.task_id, prof_name, idx)
                    actions_taken.append(f"Re-added account '{target_acc}' ({target_server or target_broker}) to {prof_name}")
                    break
                else:
                    page.context.clear_cookies()
                    try:
                        page.evaluate("() => { try { localStorage.clear(); sessionStorage.clear(); } catch(e){} }")
                    except Exception:
                        pass

            browser.close()
    except Exception as e:
        return ToolResult(
            ok=True,
            summary=f"Dashi checked ShowMyTrades profiles. Action note: {str(e)[:120]}",
            data={"target_account": target_acc, "inspected_profiles": inspected_profiles, "actions": actions_taken},
        )

    if found_profile:
        summary = (
            f"[OK] Dashi found account '{target_acc}' in ShowMyTrades profile '{found_profile}'.\n"
            f"[OK] Deleted old connection from '{found_profile}'.\n"
            f"[OK] Re-added account '{target_acc}' ({target_server or target_broker}) with updated credentials!\n"
            + ("[OK] Visibility set to Published." if published
               else "[!] Could not set it to Published. Change Visibility in the account's Settings.")
        )
    else:
        summary = f"Account '{target_acc}' was not found in any of the {len(inspected_profiles)} scanned profiles."
        _step(ctx.task_id, target_acc, "find", "Find account", "failed", f"not in {len(inspected_profiles)} profile(s)")
        for key, label in (("delete", "Delete old connection"), ("add", "Add on ShowMyTrades"), ("publish", "Set Published")):
            _step(ctx.task_id, target_acc, key, label, "skipped")

    return ToolResult(ok=True, summary=summary, data={"target_account": target_acc, "found_profile": found_profile, "actions": actions_taken})


def _row_status(row) -> str:
    text = row.inner_text()
    for word in ("Published", "Private", "Unlisted"):
        if word in text:
            return word
    return "Unknown"


def _check_accounts(args: CheckAccountsArgs, ctx: ToolContext) -> ToolResult:
    """Read-only: finds each saved account on ShowMyTrades, one by one, and reports its profile and visibility."""
    cfg = load_dashi_config()
    profiles = [p for p in cfg.get("profiles", []) if p.get("email") and p.get("password")]
    saved = [a for a in cfg.get("trading_accounts", []) if a.get("account_number")]
    wanted = [n.strip() for n in args.account_numbers if n.strip()] or [a["account_number"] for a in saved]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return ToolResult(ok=False, summary="No saved accounts to check. Add them first (Add account).")
    if not profiles:
        return ToolResult(ok=False, summary="No ShowMyTrades profile has a login saved. Add one in Settings → Dashi.")

    from playwright.sync_api import sync_playwright
    from agent.desktop_env import ensure_default_desktop

    ensure_default_desktop()
    found: dict[str, tuple[str, str]] = {}  # number -> (profile name, status)
    emit(ctx.task_id, "tool_started", f"Checking {len(wanted)} account(s) across {len(profiles)} profile(s)...")
    for num in wanted:
        _step(ctx.task_id, num, "check", "Check", "pending")
    try:
        with sync_playwright() as p:
            headless = get_settings().browser_headless
            try:
                browser = p.chromium.launch(headless=headless, channel="chrome")
            except Exception:
                browser = p.chromium.launch(headless=headless, channel=get_settings().browser_channel or "chrome")
            page = browser.new_page(viewport={"width": 1366, "height": 850})

            # Profiles where accounts were last seen go first, so most checks need one login.
            known = {a.get("profile_id") for a in saved if a.get("account_number") in wanted}
            for prof in sorted(profiles, key=lambda pr: pr.get("id") not in known):
                left = [n for n in wanted if n not in found]
                if not left:
                    break
                prof_name = prof.get("name", "Profile")
                _login_profile(page, prof, ctx.task_id)
                page.goto("https://showmytrades.com/dashboard/accounts", wait_until="domcontentloaded", timeout=25000)
                page.wait_for_timeout(2500)
                for num in left:
                    _step(ctx.task_id, num, "check", "Check", "running", f"looking in {prof_name}")
                    search = page.locator('input[type="search"], input[placeholder*="search" i], '
                                          'input[placeholder*="filter" i]').first
                    if search.count() > 0:
                        search.fill(num)
                        page.wait_for_timeout(1500)
                    row = page.locator(f'tr:has-text("{num}")').first
                    if row.count() > 0:
                        status = _row_status(row)
                        found[num] = (prof_name, status)
                        remember_account_profile(num, prof.get("id", ""))
                        emit(ctx.task_id, "tool_completed", f"{num}: {status} in {prof_name}")
                        _step(ctx.task_id, num, "check", "Check", "done" if status == "Published" else "warn",
                              f"{status} · {prof_name}")
                    else:
                        _step(ctx.task_id, num, "check", "Check", "pending", f"not in {prof_name}, trying next profile")
                page.context.clear_cookies()
                try:
                    page.evaluate("() => { try { localStorage.clear(); sessionStorage.clear(); } catch(e){} }")
                except Exception:
                    pass
            browser.close()
    except Exception as e:
        if not found:
            return ToolResult(ok=False, summary=f"Error while checking accounts: {str(e)[:150]}")

    for num in wanted:
        if num not in found:
            _step(ctx.task_id, num, "check", "Check", "failed", "not found in any profile")
    lines = []
    for num in wanted:
        if num in found:
            prof_name, status = found[num]
            mark = "[OK]" if status == "Published" else "[!]"
            lines.append(f"{mark} {num}: {status} · {prof_name}")
        else:
            lines.append(f"[X] {num}: not found in any profile")
    published = sum(1 for n in wanted if found.get(n, ("", ""))[1] == "Published")
    head = f"Checked {len(wanted)} account(s): {published} Published, {len(found) - published} not Published, " \
           f"{len(wanted) - len(found)} not found."
    return ToolResult(ok=True, summary=head + "\n" + "\n".join(lines),
                      data={"accounts": {n: {"profile": f[0], "status": f[1]} for n, f in found.items()},
                            "missing": [n for n in wanted if n not in found]})


register(
    Tool(
        name="dashi.check_accounts",
        description=(
            "Dashi: READ-ONLY check. Logs in to each ShowMyTrades profile and looks up the saved trading accounts one "
            "by one, reporting which profile each is in and whether it is Published, Private or Unlisted. Changes nothing."
        ),
        input_model=CheckAccountsArgs,
        risk=Risk.low,
        run=_check_accounts,
        timeout_s=900,
    )
)

register(
    Tool(
        name="dashi.delete_account",
        description=(
            "Dashi specialized trading tool: DELETE ONLY. Scans ShowMyTrades profiles, finds which profile has the trading "
            "account, clicks the three dots (⋮), confirms deletion, and permanently deletes it. Does NOT add any account."
        ),
        input_model=DeleteAccountArgs,
        risk=Risk.medium,
        run=_delete_trading_account,
        timeout_s=300,  # scans every profile
    )
)

register(
    Tool(
        name="dashi.add_account",
        description=(
            "Dashi specialized trading tool: ADD ONLY. Adds a new trading account to ShowMyTrades using MetaTrader 5 "
            "Direct Connection with broker server, account number and investor password, then clicks Submit. Does NOT delete anything."
        ),
        input_model=AddAccountArgs,
        risk=Risk.medium,
        run=_add_trading_account,
        timeout_s=300,  # login, add, then wait for ShowMyTrades to sync before setting Published
    )
)

register(
    Tool(
        name="dashi.update_account",
        description=(
            "Dashi specialized trading tool: UPDATE (DELETE OLD + RE-ADD FRESH). Scans ShowMyTrades profiles, deletes "
            "the existing account connection, and re-adds it with fresh credentials."
        ),
        input_model=UpdateAccountArgs,
        risk=Risk.medium,
        run=_update_trading_account,
        timeout_s=600,  # may scan every profile, then waits for the re-added account to sync
    )
)

register(
    Tool(
        name="dashi.save_accounts",
        description=(
            "Dashi: SAVE trading account credentials the user types in chat (account number, investor password, "
            "broker server) into Settings → Dashi, without touching ShowMyTrades. Use when the user gives one or many "
            "accounts to keep ('ye accounts save karo', 'container mein daal do'). Later update/add/delete commands "
            "only need the account number."
        ),
        input_model=SaveAccountsArgs,
        risk=Risk.low,
        run=_save_accounts,
    )
)

register(
    Tool(
        name="dashi.list_profiles",
        description="List all configured ShowMyTrades profiles and trading account credentials stored in Dashi.",
        input_model=NoArgs,
        risk=Risk.low,
        run=_list_dashi,
    )
)

register(
    Tool(
        name="dashi.switch_agent",
        description="Switch between 'Abbas' (General desktop agent) and 'Dashi' (ShowMyTrades trading agent).",
        input_model=SwitchAgentArgs,
        risk=Risk.low,
        run=_switch_agent,
    )
)
