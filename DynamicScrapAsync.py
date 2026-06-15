import re
import json
import asyncio
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from pydantic import BaseModel
from urllib.parse import urljoin
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright


class TaskSchema(BaseModel):
    """
    Output schema definition for the task.
    
    Args:
        items (Dict[str, Any] | List[Dict[str, Any]]): A dictionary or list of dictionaries defining the output schema.
    """
    items: Dict[str, Any] | List[Dict[str, Any]]


class ScrapeTask(BaseModel):
    """
    Input task definition for scraper.
    
    Args:
        url (str): URL to start scraping from.
        search_query (Optional[str]): Search query to use. If provided, agent will search the query and open the results.
        action_instruction (Optional[str]): Action to perform after search query. If provided, agent will perform the action.
        extract_instruction (str): Instruction to extract data from the web page.
        output_schema (TaskSchema): Output schema definition for the task.
        required_fields (Optional[List[str]]): Fields that are required to be present in the extracted data.
        max_pages (int): Maximum number of pages to scrape.
        max_repairs (int): Maximum number of repairs to attempt if extraction fails.
        total_timeout_sec (int): Total timeout for the task in seconds.
    """
    url: str
    search_query: Optional[str] = None
    action_instruction: Optional[str] = None
    extract_instruction: str
    output_schema: TaskSchema
    required_fields: Optional[List[str]] = None
    max_pages: int = 1
    max_repairs: int = 2
    total_timeout_sec: int = 300


class DynamicScrapeAgent:
    """
    An autonomous, LLM-driven web scraping agent built with Playwright and an OpenAI-compatible API.
    
    This agent uses a Large Language Model (LLM) to dynamically understand, navigate, and extract
    data from web pages without requiring hardcoded DOM selectors.

    Args:
        model (str): The model to use for the agent.
        client (AsyncOpenAI): An initialized AsyncOpenAI client instance.
        task (Dict[str, Any]): The task to perform.
        headless (bool): Whether to run the browser in headless mode. Default is False.
    
    Raises:
        ValueError: If the model is not a non-empty string.
        ValueError: If the client is not an AsyncOpenAI instance.
    
    Key Capabilities:
    - `Autonomous Navigation`: Translates natural language instructions into Playwright actions 
      (clicking, typing, scrolling, waiting) to bypass modals, execute searches, and reach target data.
    - `Intelligent Extraction`: Uses CSS-based chunking and LLM interpretation to identify product cards,
      or repeating elements, mapping them to a user-defined JSON schema.
    - `Anti-Bot Resilience`: Implements human-like interactions (preferring UI clicks over direct URL 
      navigation) to bypass simple bot protections.
    - `Self-Healing Validation`: Automatically verifies extracted data against required fields and 
      generates repair plans if the initial extraction fails or misses critical information.
    """

    # Mapping of alternative field names LLMs use → canonical run_step field names.
    _STEP_FIELD_ALIASES: Dict[str, str] = {
        "text": "value",
        "query": "value",
        "locator": "target",
        "element": "target",
        "distance": "value",
        "selector": "target",
        "timeout": "timeout_ms",
        "direction": "_direction",
    }

    # Valid standalone action names the executor understands.
    _VALID_ACTIONS = {
        "hover", "click", "double_click", "right_click", "type", 
        "select", "check", "uncheck", "wait", "wait_for", "scroll", "press",
        "navigate", "back", "forward", "refresh", "switch_tab", "close_tab", "drag",
    }

    def __init__(
            self, 
            *, 
            model: str, 
            client: AsyncOpenAI, 
            task: Dict[str, Any], 
            headless: bool = False
        ) -> None:

        if not isinstance(client, AsyncOpenAI):
            raise ValueError("client must be an AsyncOpenAI instance.")
        
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string.")

        self.page = None
        self.task = task
        self.model = model
        self.browser = None
        self.client = client
        self.playwright = None
        self.headless = headless
        self.logs: List[str] = []
        self.executed_steps: List[Dict[str, Any]] = []

    async def start(self) -> None:
        """
        Start the browser.

        Args:
            headless (bool): Whether to run the browser in headless mode.
        
        Raises:
            ValueError: If the browser is already running.
        """
        if self.browser:
            raise ValueError("Browser is already running.")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.page = await self.browser.new_page()

    async def close(self) -> None:
        """
        Close the browser and clean up resources.
        """
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
        self.page = None

    async def ask_ai(self, prompt: str, temperature: float = 0) -> str:
        """
        Ask the LLM a question.

        Args:
            prompt (str): The question to ask.
            temperature (float): The temperature to use for the LLM.
        
        Returns:
            str: The answer from the LLM.
        
        Raises:
            ValueError: If the LLM is not initialized.
            ValueError: If the LLM returns an error.
        """
        if not self.client:
            raise ValueError("LLM is not initialized.")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    async def open(self, url: str) -> None:
        """
        Open the browser to the given URL.

        Args:
            url (str): The URL to open.
        
        Raises:
            ValueError: If the browser is not running.
        """
        if not self.browser:
            raise ValueError("Browser is not running.")
        self.logs.append(f"open: {url}")
        await self.page.goto(url, wait_until="domcontentloaded")
        await self.page.wait_for_timeout(1500)

    async def safe_json_loads(self, text: str, fallback: Any) -> Any:
        """
        Safely parse JSON from a string, with fallback support.
        
        Args:
            text (str): String to parse.
            fallback (Any): Fallback value to return if parsing fails.
        """
        if not text:
            return fallback
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try extracting first JSON object/array from noisy output.
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if not m:
            return fallback
        try:
            return json.loads(m.group(1))
        except Exception:
            return fallback

    async def run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take a single action as per step and return the result.

        Args:
            step (Dict[str, Any]): The step to run.
        
        Returns:
            Dict[str, Any]: The result of the step.
        """
        action = str(step.get("action") or "").lower().strip()
        target = str(step.get("target") or "").strip()
        target_strategy = str(
            step.get("target_strategy") or 
            "text"
        ).lower().strip()
        value = str(
            step.get("value") if step.get("value") is not None else ""
        ).strip()
        timeout = int(step.get("timeout_ms") or 5000)

        result = {
            "ok": True,
            "error": "",
            "value": value,
            "action": action,
            "target": target,
        }
        self.executed_steps.append(step)

        try:
            if target and target_strategy in {"", "text"}:
                if target.startswith("//") or target.startswith("(//"):
                    target_strategy = "xpath"
                elif any(ch in target for ch in [
                    "#", ".", "[", "]", ">", ":", "="
                ]):
                    target_strategy = "css"

            def locator_from_strategy():
                if target_strategy == "role_text":
                    return self.page.get_by_role(
                        "link", 
                        name=target, 
                        exact=False
                    ).first
                if target_strategy == "label":
                    return self.page.get_by_label(target, exact=False).first
                if target_strategy == "placeholder":
                    return self.page.get_by_placeholder(target, exact=False).first
                if target_strategy == "css":
                    return self.page.locator(target).first
                if target_strategy == "xpath":
                    return self.page.locator(f"xpath={target}").first
                return self.page.get_by_text(target, exact=False).first

            async def try_multi_locator_hover_or_click(kind: str):
                attempts = []
                if target:
                    attempts.append(("declared", locator_from_strategy()))
                    attempts.append(("text_exact", self.page.get_by_text(
                        target, 
                        exact=True
                    ).first))
                    attempts.append(("text_partial", self.page.get_by_text(
                        target, 
                        exact=False
                    ).first))
                    attempts.append(("role_link", self.page.get_by_role(
                        "link", 
                        name=target, 
                        exact=False
                    ).first))
                    attempts.append(("role_button", self.page.get_by_role(
                        "button", name=target, exact=False
                    ).first))
                    if target.startswith("//") or target.startswith("(//"):
                        attempts.append((
                            "xpath_auto", 
                            self.page.locator(f"xpath={target}").first
                        ))
                    if any(ch in target for ch in ["#", ".", "[", "]", ">", ":", "="]):
                        attempts.append(("css_auto", self.page.locator(target).first))

                last_err = None
                for idx, (_, loc) in enumerate(attempts):
                    # Use the full timeout for the primary attempt, short timeout for fallbacks
                    current_timeout = timeout if idx == 0 else min(timeout, 500)
                    try:
                        if kind == "hover":
                            await loc.hover(force=True, timeout=current_timeout)
                        elif kind == "click":
                            await loc.click(timeout=current_timeout)
                        elif kind == "double_click":
                            await loc.dblclick(timeout=current_timeout)
                        elif kind == "right_click":
                            await loc.click(button="right", timeout=current_timeout)
                        return
                    except Exception as e:
                        last_err = e
                        continue

                if not attempts:
                    raise ValueError(f"Action '{kind}' requires a non-empty target.")
                if last_err:
                    raise last_err

            if action == "hover":
                await try_multi_locator_hover_or_click("hover")
            elif action == "click":
                await try_multi_locator_hover_or_click("click")
            elif action == "double_click":
                await try_multi_locator_hover_or_click("double_click")
            elif action == "right_click":
                await try_multi_locator_hover_or_click("right_click")
            elif action == "type":
                # If target looks like a CSS selector, use it directly.
                if target and any(ch in target for ch in ["#", ".", "[", "]", ">"]):  
                    locator = self.page.locator(target).first
                else:
                    locator = self.page.get_by_role("textbox", name=target).first
                    if await locator.count() == 0:
                        locator = self.page.get_by_role("searchbox", name=target).first
                    if await locator.count() == 0:
                        safe_t = target.replace('"', '\\"')
                        locator = self.page.locator(
                            f'input[aria-label*="{safe_t}" i], '
                            f'textarea[aria-label*="{safe_t}" i], '
                            f'input[placeholder*="{safe_t}" i], '
                            f'textarea[placeholder*="{safe_t}" i]'
                        ).first
                    if await locator.count() == 0:
                        locator = self.page.get_by_label(target, exact=False).first
                    if await locator.count() == 0:
                        locator = self.page.get_by_placeholder(target, exact=False).first
                    if await locator.count() == 0:
                        locator = self.page.locator(
                            "input[type='search'], input[name='q'], input[type='text'], textarea"
                        ).first
                await locator.click(timeout=timeout)
                await locator.fill(value, timeout=timeout)
            elif action == "press":
                key = value or "Enter"
                await self.page.keyboard.press(key)
            elif action == "select":
                locator = locator_from_strategy()
                await locator.select_option(label=value, timeout=timeout)
            elif action == "check":
                locator = locator_from_strategy()
                await locator.check(timeout=timeout)
            elif action == "uncheck":
                locator = locator_from_strategy()
                await locator.uncheck(timeout=timeout)
            elif action == "wait":
                ms = int(value) if value.isdigit() else 1500
                await self.page.wait_for_timeout(ms)
            elif action == "wait_for":
                wait_mode = value or "domcontentloaded"
                if wait_mode in {"load", "domcontentloaded", "networkidle"}:
                    await self.page.wait_for_load_state(wait_mode, timeout=timeout)
                else:
                    locator = locator_from_strategy()
                    await locator.wait_for(state="visible", timeout=timeout)
            elif action == "scroll":
                await self.page.mouse.wheel(0, int(value or 1400))
                await self.page.wait_for_timeout(1000)
            elif action == "navigate":
                await self.page.goto(value or target, wait_until="domcontentloaded")
            elif action == "back":
                await self.page.go_back(wait_until="domcontentloaded")
            elif action == "forward":
                await self.page.go_forward(wait_until="domcontentloaded")
            elif action == "refresh":
                await self.page.reload(wait_until="domcontentloaded")
            elif action == "switch_tab":
                pages = self.browser.contexts[0].pages if self.browser and self.browser.contexts else []
                idx = int(value or 0)
                if idx < 0 or idx >= len(pages):
                    raise ValueError(f"tab index out of range: {idx}")
                self.page = pages[idx]
                await self.page.bring_to_front()
            elif action == "close_tab":
                await self.page.close()
                pages = self.browser.contexts[0].pages if self.browser and self.browser.contexts else []
                if not pages:
                    raise ValueError("no tab left after close_tab")
                self.page = pages[-1]
                await self.page.bring_to_front()
            elif action == "drag":
                payload = await self.safe_json_loads(value, fallback={})
                source = payload.get("source", "")
                target_drop = payload.get("target", "")
                if not source or not target_drop:
                    raise ValueError(
                        "drag action requires value JSON with source and target"
                    )
                src = self.page.get_by_text(source, exact=False).first
                dst = self.page.get_by_text(target_drop, exact=False).first
                await src.drag_to(dst, timeout=timeout)
            else:
                # Normalize common LLM-invented action aliases before failing.
                # LLMs sometimes hallucinate action names not in the spec.
                ACTION_ALIASES = {
                    # type variants
                    "type_text": "type",
                    "input_text": "type",
                    "enter_text": "type",

                    # press variants
                    "key_press": "press",
                    "press_key": "press",

                    # hover variants
                    "mouse_over": "hover",
                    "hover_element": "hover",

                    # click variants
                    "click_link": "click",
                    "click_button": "click",
                    "click_element": "click",

                    # navigation variants
                    "go_back": "back",
                    "reload": "refresh",
                    "go_forward": "forward",
                    "page_refresh": "refresh",

                    # scroll variants
                    "scroll_to": "scroll",
                    "scroll_down": "scroll",
                    "scroll_page": "scroll",
                    "scroll_to_element": "scroll",

                    # wait variants
                    "wait_for_element": "wait_for",
                    "wait_for_selector": "wait_for",
                    "wait_for_visible_element": "wait_for",
                }
                
                aliased = ACTION_ALIASES.get(action)
                if aliased:
                    canonical_step = dict(step)
                    canonical_step["action"] = aliased
                    self.logs.append(f"action_alias: '{action}' → '{aliased}'")
                    return await self.run_step(canonical_step)
                
                result["ok"] = False
                result["error"] = f"Unsupported action: {action}"
                return result

            await self.page.wait_for_timeout(1200)
            return result
        except Exception as e:
            result["ok"] = False
            result["error"] = str(e)
            return result

    async def _collect_visible_inputs(self) -> List[Dict[str, str]]:
        """
        Pure data collector: JS dumps every visible input element's raw
        attributes as a list of dicts. No scoring, no decisions — just data.
        The LLM will decide what to do with it.

        Returns:
            List[Dict[str, str]]: List of visible input elements.
        
        Raises:
            ValueError: If the browser is not running.
        """
        return await self.page.evaluate("""
() => {
    const isVisible = (el) => {
        const s = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        return s.display !== 'none' && s.visibility !== 'hidden'
            && s.opacity !== '0' && r.width > 4 && r.height > 4;
    };
    const uniqueSel = (el) => {
        if (el.id) return '#' + CSS.escape(el.id);
        const name = el.getAttribute('name');
        if (name) return 'input[name="' + CSS.escape(name) + '"]';
        let path = el.tagName.toLowerCase();
        let cur = el;
        for (let i = 0; i < 3 && cur.parentElement; i++) {
            cur = cur.parentElement;
            path = cur.tagName.toLowerCase() + ' > ' + path;
        }
        return path;
    };
    const form_action = (el) => {
        const f = el.closest('form');
        return f ? (f.getAttribute('action') || '') : '';
    };
    return Array.from(document.querySelectorAll(
        'input, textarea, [role="searchbox"], [role="combobox"]'
    ))
    .filter(isVisible)
    .map(el => ({
        selector:    uniqueSel(el),
        tag:         el.tagName.toLowerCase(),
        type:        el.getAttribute('type')        || '',
        name:        el.getAttribute('name')        || '',
        id:          el.id                         || '',
        placeholder: el.getAttribute('placeholder') || '',
        aria_label:  el.getAttribute('aria-label')  || '',
        title:       el.getAttribute('title')       || '',
        role:        el.getAttribute('role')        || '',
        class_list:  el.className                  || '',
        form_action: form_action(el),
        width:       Math.round(el.getBoundingClientRect().width),
    }));
}""")

    async def _llm_pick_search_box(self, inputs: List[Dict[str, str]]) -> Optional[str]:
        """
        Send the raw input list to the LLM and ask it to pick the search box.
        Returns the CSS selector string, or None if the LLM can't decide.

        Args:
            inputs (List[Dict[str, str]]): List of input elements.
        
        Returns:
            Optional[str]: CSS selector of the search box.
        """
        if not inputs:
            return None
        prompt = f"""
You are a browser element analyst. The page has these visible input elements:

{json.dumps(inputs, ensure_ascii=False, indent=2)}

Task: Identify which ONE element is the main search box where a user would type
a search query. Consider: type, name, placeholder, aria_label, role, id, class_list,
form_action, and width (wider inputs are more likely to be the main search bar).

Return ONLY valid JSON, no markdown:
{{
  "selector": "<the exact selector string from the list above>",
  "reason": "<one short sentence>"
}}

If you cannot identify a search box, return:
{{"selector": null, "reason": "no search box found"}}
"""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback={})
        sel = parsed.get("selector") if isinstance(parsed, dict) else None
        reason = parsed.get("reason", "") if isinstance(parsed, dict) else ""
        self.logs.append(f"llm_search_box: sel={sel!r} reason={reason!r}")
        return sel if isinstance(sel, str) and sel.strip() else None

    async def detect_search_and_run(self, query: str) -> Dict[str, Any]:
        """
        Fully LLM-driven search:
        1. JS collects raw attributes of all visible inputs (no scoring).
        2. LLM reads that data and picks the right search box selector.
        3. Agent types the query and presses Enter.
        Works on any website with zero hardcoded selectors or heuristics.

        Args:
            query (str): The search query.
        
        Returns:
            Dict[str, Any]: The result of the search.
        """
        diagnostics = {"ok": True, "reason": "", "query": query}
        try:
            # Step 1: collect raw DOM data — JS is only a camera, not a brain.
            inputs = await self._collect_visible_inputs()
            if not inputs:
                raise RuntimeError("No visible input elements found on page.")

            # Step 2: LLM decides which input is the search box.
            sel = await self._llm_pick_search_box(inputs)
            if not sel:
                raise RuntimeError("LLM could not identify a search box on this page.")

            # Step 3: interact.
            box = self.page.locator(sel).first
            await box.wait_for(state="visible", timeout=6000)
            await box.click(timeout=5000)
            await box.fill("", timeout=3000)   # clear any pre-filled text
            await box.press_sequentially(query, delay=45)    # human-like typing
            await self.page.keyboard.press("Enter")
            await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
            await self.page.wait_for_timeout(2500)
            return diagnostics
        except Exception as e:
            diagnostics["ok"] = False
            diagnostics["reason"] = str(e)
            return diagnostics

    async def collect_visible_links(self) -> List[Dict[str, str]]:
        """
        Collect visible links from the page.

        Returns:
            List[Dict[str, str]]: List of visible links.
        """
        items = await self.page.evaluate("""
() => {
    const nodes = Array.from(document.querySelectorAll('a[href]'));
    const isVisible = (el) => {
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
    };
    const out = [];
    for (const a of nodes) {
    if (!isVisible(a)) continue;
    const name = (a.textContent || '').replace(/\\s+/g, ' ').trim();
    const href = a.getAttribute('href') || '';
    if (!name || !href) continue;
    if (href.startsWith('#') || href.startsWith('javascript:')) continue;
    out.push({ name, url: new URL(href, location.origin).href });
    }
    return out;
}"""
        )
        dedup = {}
        for x in items:
            dedup[f"{x['name']}|{x['url']}"] = x
        return list(dedup.values())

    async def required_action_types(self, action_instruction: str) -> List[str]:
        """
        Ask LLM which action types the instruction requires.
        Replaces the old hardcoded keyword matching.

        Args:
            action_instruction (str): The action instruction to classify.
        
        Returns:
            List[str]: List of action types.
        """
        if not action_instruction.strip():
            return []
        prompt = f"""
You are a browser action classifier.
Given this action instruction, list ALL action types it requires.

Instruction: {action_instruction}

Return ONLY valid JSON array of strings from this set:
["hover", "click", "type", "scroll", "press", "navigate", "wait"]

Examples:
- "Hover on Software Solutions" → ["hover"]
- "Click Sign In and type credentials" → ["click", "type"]
- "Search for products" → ["type"]
- "Open the menu and click Settings" → ["click"]

Return [] if no specific action is required.
No markdown. JSON array only."""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback=[])
        if isinstance(parsed, list):
            return [str(x).lower().strip() for x in parsed if isinstance(x, str)]
        return []

    async def normalize_steps(self, steps: Any) -> List[Dict[str, Any]]:
        """
        Normalize a list of raw step dicts from the LLM.
        - Remaps alternative field names (locator→target, distance→value, etc.)
        - Splits compound action strings like 'scroll|wait' or 'hover/click'
          into individual sequential steps.
        - Drops steps without an action key.
        
        Args:
            steps (Any): List of raw step dicts from the LLM.
        
        Returns:
            List[Dict[str, Any]]: List of normalized step dicts.
        """
        if not isinstance(steps, list):
            return []

        normalized = []
        for s in steps:
            if not isinstance(s, dict) or not s.get("action"):
                continue

            step = dict(s)  # copy so we don't mutate the original
            for alias, canonical in self._STEP_FIELD_ALIASES.items():
                if alias in step and canonical not in step:
                    step[canonical] = step.pop(alias)
                elif alias in step:  # both present, just remove alias
                    step.pop(alias)

            # Split compound action strings (e.g. "scroll|wait", "hover/click").
            raw_action = (step.get("action") or "").strip()
            parts = [p.strip() for p in re.split(r"[|/,+]", raw_action) if p.strip()]
            if len(parts) > 1:
                # Emit one step per sub-action, inheriting target/value from original.
                for part in parts:
                    sub = dict(step)
                    sub["action"] = part
                    normalized.append(sub)
            else:
                normalized.append(step)
        return normalized

    async def validate_planned_steps(
            self, steps: List[Dict[str, Any]], required_actions: List[str]
        ) -> Dict[str, Any]:
        """
        Validate the planned steps.

        Args:
            steps (List[Dict[str, Any]]): List of planned steps.
            required_actions (List[str]): List of required action types.
        
        Returns:
            Dict[str, Any]: Dictionary containing the validation result.
        """
        # 'wait' and 'press' are implementation details — the planner may achieve
        _OPTIONAL_ACTIONS = {"wait", "press", "wait_for"}
        enforced = [ra for ra in required_actions if ra not in _OPTIONAL_ACTIONS]

        if enforced and not steps:
            return {
                "ok": False, 
                "reason": "Planner returned empty steps for non-empty action_instruction."
            }

        action_types = [(s.get("action") or "").strip().lower() for s in steps]
        for ra in enforced:
            if ra not in action_types:
                return {
                    "ok": False, 
                    "reason": f"Planner missed required action: {ra}"
                }

        return {"ok": True, "reason": ""}

    async def check_task_possibility(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the task is possible.

        Args:
            task (Dict[str, Any]): The task to check.
        
        Returns:
            Dict[str, Any]: Dictionary containing the possibility result.
        """
        prompt = f"""
You are a browser scraping feasibility checker.
Return ONLY JSON:
{{
  "possible": true,
  "reason": "short reason"
}}

Task:
{json.dumps(task, ensure_ascii=False)}

Rules:
- possible=true when task can likely be done with browser actions + visible page data extraction.
- possible=false only when fundamentally blocked (e.g. requires login you cannot complete, captcha gate, data not on page, impossible external dependency).
- Default to possible=true. Do not reject tasks merely because the target domain (like LinkedIn or Amazon) generally requires login, unless the task explicitly requires an authenticated session.
- Keep reason short."""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback={})
        if isinstance(parsed, dict) and isinstance(parsed.get("possible"), bool):
            return {
                "possible": parsed["possible"], 
                "reason": str(parsed.get("reason", "")).strip()
            }
        return {"possible": True, "reason": "No clear blocker detected."}

    async def plan_actions(
            self, task: Dict[str, Any], planner_feedback: str = ""
        ) -> Dict[str, Any]:
        """
        Plan the actions to take.

        Args:
            task (Dict[str, Any]): The task to plan.
            planner_feedback (str): Feedback from the previous planner.
        
        Returns:
            Dict[str, Any]: Dictionary containing the planned steps.
        """
        prompt = f"""
You are a browser action planner.

Return ONLY valid JSON object with this exact shape:
{{
  "steps": [
    {{
      "action": "hover|click|type|press|wait|scroll",
      "target": "...",
      "value": "...",
      "timeout_ms": 5000
    }}
  ],
  "reasoning": "short"
}}

Allowed actions:
- hover (target)
- click (target)
- double_click (target)
- right_click (target)
- type (target, value)
- press (value)
- select (target, value)
- check (target)
- uncheck (target)
- wait (value in ms)
- wait_for (value can be: load|domcontentloaded|networkidle, or use target for element wait)
- scroll (value in px)
- navigate (value=url or target=url)
- back
- forward
- refresh
- switch_tab (value=index)
- close_tab
- drag (value as JSON string: {{"source":"...","target":"..."}})

Task:
{json.dumps(task, ensure_ascii=False)}

Validation feedback from previous attempt:
{planner_feedback or "none"}

Rules:
- Never return empty "steps" if action_instruction is non-empty.
- The first step must directly execute the user action intent.
- If instruction explicitly asks an action, plan MUST include that action type..
- Use visible UI text for target whenever possible.
- IMPORTANT: Prefer interacting with the page (clicking buttons/links, typing in search bars) instead of using the `navigate` action to jump directly to a new URL. Direct navigation on many sites triggers anti-bot login walls, whereas clicking UI elements bypasses them.
- Keep steps minimal and deterministic.
- No markdown. No text outside JSON."""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback={})
        if isinstance(parsed, dict):
            steps = await self.normalize_steps(parsed.get("steps", []))
            return {"steps": steps, "raw": raw}
        # backward compatibility if model still returns array
        steps = await self.normalize_steps(parsed)
        return {"steps": steps, "raw": raw}

    async def fallback_steps_from_instruction(self, instruction: str) -> List[Dict[str, Any]]:
        """
        LLM-driven fallback step generator.
        Called only when plan_actions fails twice. Returns minimal deterministic steps.
        
        Args:
            instruction (str): The instruction to generate steps from.
        
        Returns:
            List[Dict[str, Any]]: List of fallback steps.
        """
        if not (instruction or "").strip():
            return []
        prompt = f"""
You are a browser action planner generating a MINIMAL fallback plan.
Return ONLY valid JSON array of steps:
[
  {{
      "action": "hover|click|type|scroll|wait|press", 
      "target": "...", 
      "value": "...", 
      "timeout_ms": 7000
  }}
]

Instruction: {instruction}

Rules:
- 1-2 steps maximum.
- Use visible text for target (e.g. "Software Solutions", "Sign In").
- Prioritize clicking links/buttons over direct `navigate` actions to bypass login walls.
- No markdown. JSON array only.
- If instruction is unclear, return []."""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback=[])
        return await self.normalize_steps(parsed) if isinstance(parsed, list) else []

    async def extract_hover_target_from_instruction(self, instruction: str) -> str:
        """
        LLM extracts the hover target text from any natural-language instruction.
        
        Args:
            instruction (str): The instruction to extract the hover target from.
        
        Returns:
            str: The hover target text.
        """
        if not (instruction or "").strip():
            return ""
        prompt = f"""
Extract the hover target (the UI element to hover over) from this instruction.
Return ONLY the visible text label to hover on, nothing else.
If no hover target found, return empty string.

Instruction: {instruction}

Examples:
- "Hover on Software Solutions in the header." → Software Solutions
- "Mouse over the Products menu" → Products
- "Click the login button" → (empty, this is a click not hover)"""
        raw = await self.ask_ai(prompt)
        return raw.strip().strip('"\' ') if raw else ""

    async def map_to_schema(
            self, 
            instruction: str, 
            schema: Dict[str, Any], 
            source_rows: List[Dict[str, Any]]
        ) -> Optional[List[Dict[str, Any]]]:
        """
        Map the extracted data to the schema.

        Args:
            instruction (str): The instruction to map the data to.
            schema (Dict[str, Any]): The schema to map the data to.
            source_rows (List[Dict[str, Any]]): The source rows to map the data to.
        
        Returns:
            Optional[List[Dict[str, Any]]]: The mapped data.
        """
        prompt = f"""
Extract data based on user instruction and schema.
Return ONLY valid JSON array.

Instruction:
{instruction}

Schema:
{json.dumps(schema, ensure_ascii=False)}

Rows:
{json.dumps(source_rows, ensure_ascii=False)}

Rules:
- Do not invent facts.
- Filter the rows based on the instruction. Extract ONLY items that match the user's intent, and discard irrelevant items (e.g., standard navigation, footers, sign-in links).
- Use empty string for missing fields.
- Extract ALL valid items present in the rows. Do not stop early!
- Return [] if nothing matches."""
        raw = await self.ask_ai(prompt)
        data = await self.safe_json_loads(raw, fallback=None)
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            return data["items"]
        if isinstance(data, list):
            return data
        return None

    async def validate_output(
            self, data: List[Dict[str, Any]], required_fields: List[str]
        ) -> Dict[str, Any]:
        """
        Validate the extracted data.

        Args:
            data (List[Dict[str, Any]]): The extracted data.
            required_fields (List[str]): List of required fields.
        
        Returns:
            Dict[str, Any]: Dictionary containing the validation result.
        """
        if not data:
            return {"ok": False, "reason": "Empty extracted data"}
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                return {"ok": False, "reason": f"Row {i} is not an object"}
            for f in required_fields:
                v = row.get(f, "")
                if not isinstance(v, str) or not v.strip():
                    return {
                        "ok": False, 
                        "reason": f"Row {i} missing required field: {f}"
                    }
        
        return {"ok": True, "reason": ""}

    async def choose_source_scope(
        self,
        task: Dict[str, Any],
        links: List[Dict[str, str]],
    ) -> str:
        """
        Choose best extraction source for this task.

        Args:
            task (Dict[str, Any]): The task to choose the source for.
            links (List[Dict[str, str]]): List of links to choose the source from.
        
        Returns:
            str: The chosen source scope.
        """
        prompt = f"""
Choose best extraction source for this task.
Return ONLY JSON:
{{
    "scope": "links|combined",
    "reason": "short"
}}

Task:
{json.dumps(task, ensure_ascii=False)}

Links sample:
{json.dumps(links[:40], ensure_ascii=False)}"""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback={})
        scope = parsed.get("scope") if isinstance(parsed, dict) else ""
        if scope in {"links", "combined"}:
            return scope
        return "links"

    async def generate_extraction_spec(
        self,
        task: Dict[str, Any],
        links: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Generate extraction spec for the task.

        Args:
            task (Dict[str, Any]): The task to generate the spec for.
            links (List[Dict[str, str]]): List of links to choose the source from.
        
        Returns:
            Dict[str, Any]: Dictionary containing the extraction spec.
        """
        # Extract field names directly from the schema so the spec uses
        schema_items = task.get("output_schema", {}).get("items", {})
        if isinstance(schema_items, dict):
            schema_field_names = list(schema_items.keys())
        elif isinstance(schema_items, list) and schema_items and isinstance(schema_items[0], dict):
            schema_field_names = list(schema_items[0].keys())
        else:
            schema_field_names = task.get("required_fields", [])
        
        required_fields_hint = (
            f"IMPORTANT: Use EXACTLY these field names (matching the schema): "
            f"{json.dumps(schema_field_names)}\n"
            if schema_field_names else ""
        )

        prompt = f"""
You are an extraction-spec agent for browser scraping.

Return ONLY valid JSON with this exact shape:
{{
  "container_selectors": ["..."],
  "skip_items": <integer, if task instruction asks to skip items (e.g. 'skip first 10' -> 10), else 0>,
  "max_items": <integer, if task instruction asks for a specific limit (e.g. 'first 10' -> 10). If no limit is asked, use 200>,
  "fields": [
    {{
      "name": "field_name",
      "selector_candidates": ["..."],
      "attr": "text|href|src|aria-label|data-*",
      "required": true
    }}
  ],
  "fallback_mode": "links_only|container_text"
}}

Task:
{json.dumps(task, ensure_ascii=False)}

Links sample:
{json.dumps(links[:25], ensure_ascii=False)}

Rules:
{required_fields_hint}- Use CSS selectors only.
- Keep selectors short and robust.
- If URL is needed, include href-capable selector candidate.
- No prose, no markdown, JSON only."""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback={})
        if not isinstance(parsed, dict):
            return {}
        return parsed

    async def validate_extraction_spec(self, spec: Dict[str, Any]) -> bool:
        """
        Validate the extraction spec.

        Args:
            spec (Dict[str, Any]): The extraction spec to validate.
        
        Returns:
            bool: True if the spec is valid, False otherwise.
        """
        if not isinstance(spec, dict):
            return False
        fields = spec.get("fields")
        if not isinstance(fields, list) or not fields:
            return False
        for f in fields:
            if not isinstance(f, dict):
                return False
            if not isinstance(f.get("name"), str) or not f["name"].strip():
                return False
            sels = f.get("selector_candidates", [])
            if not isinstance(sels, list) or not sels:
                return False
            if any(not isinstance(s, str) or not s.strip() for s in sels):
                return False
        return True

    async def collect_by_dynamic_spec(
            self, spec: Dict[str, Any], task: Dict[str, Any] = None
        ) -> List[Dict[str, Any]]:
        """
        Collect data based on the extraction spec.

        Args:
            spec (Dict[str, Any]): The extraction spec to collect data based on.
            task (Dict[str, Any]): The task to collect data based on.
        
        Returns:
            List[Dict[str, Any]]: List of collected data.
        """
        container_selectors = spec.get("container_selectors", [])
        if not isinstance(container_selectors, list):
            container_selectors = []
        fields = spec.get("fields", [])
        max_items = int(spec.get("max_items") or 200)
        fallback_mode = (spec.get("fallback_mode") or "container_text").strip()
        
        # Check if the task instruction asks to ignore ads/promotions
        ignore_ads = False
        if task:
            extract_instr = task.get("extract_instruction", "").lower()
            if ("ignore" in extract_instr or "skip" in extract_instr) and (
                "ad" in extract_instr or 
                "promot" in extract_instr or 
                "sponsor" in extract_instr
            ):
                ignore_ads = True

        # Step 1: Use JS strictly to find VISIBLE containers (since bs4 can't check computed visibility)
        # and extract their outerHTML.
        payload = await self.page.evaluate("""
({ containerSelectors, maxItems }) => {
    const isVisible = (el) => {
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
    };
    const safeQueryAll = (root, sel) => {
    try { return Array.from(root.querySelectorAll(sel)); } catch { return []; }
    };

    let containers = [];
    for (const sel of containerSelectors || []) {
    const found = safeQueryAll(document, sel);
    if (found.length >= 2) { containers = found; break; }
    if (!containers.length && found.length) containers = found;
    }
    if (!containers.length) {
    containers = safeQueryAll(document, 'article, li, tr, .card, .item, .product, section');
    }
    containers = containers.filter(isVisible).slice(0, 500);

    const allLinks = Array.from(document.querySelectorAll('a[href]')).filter(isVisible).map(a => ({
        text: (a.textContent || '').trim(),
        href: a.getAttribute('href') || ''
    }));

    return {
        base_url: document.baseURI || location.href,
        htmls: containers.map(c => c.outerHTML),
        all_links: allLinks
    };
}
        """,
            {
                "containerSelectors": container_selectors,
                "maxItems": max_items,
            },
        )

        base_url = payload.get("base_url", "")
        htmls = payload.get("htmls", [])
        all_links = payload.get("all_links", [])

        rows = []
        # Step 2: Use BeautifulSoup in Python for robust parsing and data extraction.
        for html in htmls:
            soup = BeautifulSoup(html, "html.parser")

            # Ad/Sponsored Filter: skip containers that are explicitly marked as sponsored
            text_lower = soup.get_text(separator=" ", strip=True).lower()
            if ignore_ads:
                # But be careful not to skip everything. Only skip if 'sponsored' is prominent or explicitly asked to ignore
                if "sponsored" in text_lower.split() or "promoted" in text_lower.split():
                    continue

            row = {}
            for f in fields:
                val = ""
                attr = str(f.get("attr") or "text").lower()
                fname = str(f.get("name") or "").lower()

                # Try LLM-provided CSS selector candidates
                for sel in f.get("selector_candidates", []):
                    try:
                        el = soup.select_one(sel)
                        if el:
                            if attr in ("text", ""):
                                val = el.get_text(separator=" ", strip=True)
                            elif attr in ("href", "src"):
                                raw_attr = el.get(attr)
                                if raw_attr:
                                    val = urljoin(base_url, raw_attr)
                            else:
                                val = el.get(attr) or ""
                                if isinstance(val, list):
                                    val = " ".join(val)

                            if val:
                                break
                    except Exception:
                        continue # Invalid selector syntax

                # Smart fallback logic via BeautifulSoup (regex & tag matching)
                if not val:
                    is_title_field = attr in ("text", "") and any(
                        k in fname for k in ["title", "name", "heading", "label"]
                    )
                    is_url_field = attr == "href" or any(
                        k in fname for k in ["url", "link", "href"]
                    )

                    if is_title_field:
                        heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                        if heading:
                            val = heading.get_text(separator=" ", strip=True)
                        else:
                            # Fallback to any element with a title-like class using bs4 regex matching
                            title_el = soup.find(class_=re.compile(r"title|name|heading", re.I))
                            if title_el:
                                val = title_el.get_text(separator=" ", strip=True)
                            else:
                                # Last resort: use the container's entire text if it's reasonably short
                                txt = soup.get_text(separator=" ", strip=True)
                                if txt and len(txt) < 150:
                                    val = txt

                    if is_url_field:
                        # Find the first meaningful anchor link
                        for a in soup.find_all("a", href=True):
                            href = a["href"]
                            if href and not href.startswith(("#", "javascript:")):
                                val = urljoin(base_url, href)
                                break
                
                row[f["name"]] = val or ""

            if any(row.values()):
                rows.append(row)

        if not rows and fallback_mode == "links_only":
            for a in all_links[:max_items]:
                if a["text"] or a["href"]:
                    rows.append({
                        "title": re.sub(r"\s+", " ", a["text"]),
                        "url": urljoin(base_url, a["href"])
                    })
            
        if not rows and fallback_mode == "container_text":
            for html in htmls:
                soup = BeautifulSoup(html, "html.parser")
                # Remove noise tags before extracting text
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                if text:
                    rows.append({"raw_text": text[:500]})

        return rows

    async def build_source_rows(
        self,
        task: Dict[str, Any],
        links: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Build source rows for the task.

        Args:
            task (Dict[str, Any]): The task to build source rows for.
            links (List[Dict[str, str]]): List of links to build source rows from.
        
        Returns:
            List[Dict[str, Any]]: List of source rows.
        """
        # Ask LLM to classify which schema fields are title-like and URL-like.
        schema_items = task.get("output_schema", {}).get("items", {})
        schema_field_names = list(schema_items.keys())
        title_fields: set = set()
        url_fields: set = set()
        if schema_field_names:
            classify_prompt = f"""
Given these schema fields, identify which represents the display name/title of an item,
and which represents its URL/link.

Schema fields: {json.dumps(schema_field_names)}

Return ONLY JSON:
{{
  "title_fields": ["..."],
  "url_fields": ["..."]
}}

Rules:
- title_fields: fields that hold a human-readable text name (e.g. "title", "name", "author", "heading").
- url_fields: fields that hold a hyperlink/URL (e.g. "url", "product_url", "link", "permalink", "href").
- A field can appear in both if ambiguous.
- Return empty list if no match.
- JSON only, no markdown."""
            raw = await self.ask_ai(classify_prompt)
            parsed = await self.safe_json_loads(raw, fallback={})
            if isinstance(parsed, dict):
                title_fields = set(parsed.get("title_fields") or [])
                url_fields   = set(parsed.get("url_fields") or [])
            self.logs.append(
                f"field_classify: title={sorted(title_fields)} "
                f"url={sorted(url_fields)}"
            )

        async def _identify_item_links() -> List[Dict[str, str]]:
            """Ask LLM to filter visible links to only actual target items (not nav/footer/ads)."""
            if not links:
                return []
            sample = links[:60]
            prompt = f"""
Task: {task.get('extract_instruction', 'Extract items from the page.')}

These are visible links on the page:
{json.dumps(sample, ensure_ascii=False)}

Return ONLY the 0-based indices of links that are actual items to extract.
Exclude: navigation menus, header links, footer links, social media, account links, ads, department/category links.
Include: product links, article links, listing links that directly match the task.

Return ONLY a JSON array of integers: [2, 5, 8, ...]
Return [] if none match.
No markdown. No explanation."""
            raw = await self.ask_ai(prompt)
            indices = await self.safe_json_loads(raw, fallback=[])
            if not isinstance(indices, list):
                return []
            return [
                sample[i] for i in indices 
                if isinstance(i, int) and 0 <= i < len(sample)
            ]

        def _enrich_from_item_links(rows: List[Dict], item_links: List[Dict]) -> List[Dict]:
            """Fill empty title/url fields in rows using LLM-identified item links.
            Matches by position — both rows and item_links are ordered top-to-bottom.
            """
            if not item_links:
                return rows
            enriched = []
            for i, row in enumerate(rows):
                new_row = dict(row)
                if i < len(item_links):
                    pl = item_links[i]
                    for tf in title_fields:
                        if tf in new_row and not new_row[tf] and pl.get("name"):
                            new_row[tf] = pl["name"]
                    for uf in url_fields:
                        if uf in new_row and not new_row[uf] and pl.get("url"):
                            new_row[uf] = pl["url"]
                
                enriched.append(new_row)
            return enriched

        # Dynamic evaluator agent path first.
        spec = await self.generate_extraction_spec(task, links)
        self.last_extraction_spec = spec
        if await self.validate_extraction_spec(spec):
            self.logs.append("dynamic_spec: valid")
            dynamic_rows = await self.collect_by_dynamic_spec(spec, task)
            if dynamic_rows:
                # Log the actual field names so we can debug remap issues.
                sample_keys = list(dynamic_rows[0].keys()) if dynamic_rows else []
                self.logs.append(
                    f"dynamic_spec_rows: {len(dynamic_rows)} keys={sample_keys}"
                )

                # Enrich rows that still have empty title/url after CSS extraction.
                if title_fields or url_fields:
                    needs_enrich = any(
                        not row.get(tf)
                        for row in dynamic_rows
                        for tf in title_fields
                        if tf in row
                    )
                    if needs_enrich:
                        item_links = await _identify_item_links()
                        self.logs.append(
                            f"item_links: llm identified {len(item_links)} "
                            f"items from {len(links)} links"
                        )
                        enriched = _enrich_from_item_links(dynamic_rows, item_links)
                        filled = sum(
                            1 for r in enriched
                            if any(r.get(f) for f in title_fields)
                        ) - sum(
                            1 for r in dynamic_rows
                            if any(r.get(f) for f in title_fields)
                        )
                        if filled > 0:
                            self.logs.append(
                                f"link_enrich: filled titles in {filled} rows"
                            )
                        return enriched
                return dynamic_rows
            self.logs.append("dynamic_spec_rows: 0")
        else:
            self.logs.append("dynamic_spec: invalid, using fallback collectors")

        scope = await self.choose_source_scope(task, links)
        self.logs.append(f"extract_scope: {scope}")
        return links

    async def propose_repair(
        self,
        task: Dict[str, Any],
        failure_reason: str,
        observe: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Propose repair steps for the task.

        Args:
            task (Dict[str, Any]): The task to propose repair steps for.
            failure_reason (str): The reason for the failure.
            observe (Dict[str, Any]): The observed data.
        
        Returns:
            List[Dict[str, Any]]: List of repair steps.
        """
        prompt = f"""
You are a browser recovery planner.
Return ONLY JSON:
{{
  "steps": [
    {{
      "action": "hover|click|double_click|right_click|type|press|select|check|uncheck|wait|wait_for|scroll|navigate|back|forward|refresh|switch_tab|close_tab|drag",
      "target": "...",
      "target_strategy": "role_text|text|label|placeholder|css|xpath",
      "value": "...",
      "timeout_ms": 5000
    }}
  ]
}}

Task:
{json.dumps(task, ensure_ascii=False)}

Failure reason:
{failure_reason}

Observe:
{json.dumps(observe, ensure_ascii=False)}

Rules:
- Return 1-3 minimal repair steps only.
- Focus on revealing missing data (open menu, click tab, scroll, wait_for visible element).
- No explanation outside JSON."""
        raw = await self.ask_ai(prompt)
        parsed = await self.safe_json_loads(raw, fallback={})
        if isinstance(parsed, dict):
            return await self.normalize_steps(parsed.get("steps", []))
        return await self.normalize_steps(parsed)

    async def observe_failure_context(self) -> Dict[str, Any]:
        """
        Observe the failure context.

        Returns:
            Dict[str, Any]: The failure context.
        """
        return await self.page.evaluate("""
() => {
    const visibleButtons = Array.from(document.querySelectorAll('button, a[href], input[type="submit"]'))
    .filter(el => {
        const s = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
    })
    .slice(0, 80)
    .map(el => ({
        tag: el.tagName.toLowerCase(),
        text: (el.textContent || el.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim().slice(0, 120),
        href: el.getAttribute('href') || ''
    }));
    return {
    url: location.href,
    title: document.title,
    visible_navigation_like_count: visibleButtons.length,
    visible_navigation_like: visibleButtons
    };
}""")

    async def run_task(self) -> Dict[str, Any]:
        """
        Run the scrape task.

        Returns:
            Dict[str, Any]: The result of the task.
                - ok (bool): Whether the task was completed successfully.
                - data (List[Dict[str, Any]]): The extracted data.
                - diagnostics (Dict[str, Any]): The diagnostics of the task.
        """
        await self.start()
        await self.open(self.task["url"])

        feasibility = await self.check_task_possibility(self.task)
        self.logs.append(f"feasibility: {feasibility}")
        if not feasibility["possible"]:
            observe = await self.observe_failure_context()
            return {
                "ok": False,
                "data": [],
                "diagnostics": {
                    "reason": feasibility["reason"] or "Task not feasible in current run.",
                    "logs": self.logs,
                    "executed_steps": self.executed_steps,
                    "observe": observe,
                },
            }

        search_query = (self.task.get("search_query") or "").strip()
        search_was_handled = False
        if search_query:
            search_result = await self.detect_search_and_run(search_query)
            self.logs.append(f"search: {search_result}")
            search_was_handled = search_result["ok"]

        action_instruction = (self.task.get("action_instruction") or "").strip()

        # Ask LLM whether the instruction is purely a search directive (no extra actions).
        _search_only_instruction = False
        if search_was_handled and action_instruction:
            _skip_prompt = f"""
The user already ran a search successfully.
Instruction: {action_instruction}

Return ONLY JSON: {{"skip": true}} if this instruction is ONLY about performing a search or viewing/opening the search results
(no clicking menus, no hovering, no navigating to other pages, just submitting a search query and waiting for results).
Return {{"skip": false}} if it requires any additional browser actions beyond searching."""
            _skip_raw = await self.ask_ai(_skip_prompt)
            _skip_parsed = await self.safe_json_loads(_skip_raw, fallback={})
            _search_only_instruction = bool(
                _skip_parsed.get("skip")
            ) if isinstance(_skip_parsed, dict) else False
        elif not action_instruction:
            _search_only_instruction = search_was_handled

        if _search_only_instruction:
            self.logs.append(
                "planner: skipped — search already handled by detect_search_and_run"
            )
            planned_steps = []
        else:
            required_actions = await self.required_action_types(
                action_instruction
            )
            plan_result = await self.plan_actions(self.task)
            planned_steps = plan_result["steps"]
            plan_validation = await self.validate_planned_steps(
                planned_steps, 
                required_actions
            )

            # One strict replan retry with explicit validation feedback.
            if not plan_validation["ok"]:
                self.logs.append(f"planner invalid: {plan_validation['reason']}")
                replan_result = await self.plan_actions(
                    self.task, 
                    planner_feedback=plan_validation["reason"]
                )
                planned_steps = replan_result["steps"]
                plan_validation = await self.validate_planned_steps(
                    planned_steps, required_actions
                )
                if plan_validation["ok"]:
                    self.logs.append("planner: succeeded after one retry")
                else:
                    self.logs.append(
                        f"planner retry invalid: {plan_validation['reason']}"
                    )

            # Final safety fallback only after planner + retry fail.
            if not plan_validation["ok"] and action_instruction:
                planned_steps = await self.fallback_steps_from_instruction(
                    action_instruction
                )
                if planned_steps:
                    self.logs.append(
                        "planner: using llm fallback after failed planner+retry"
                    )
                else:
                    self.logs.append(
                        "planner: no fallback could be created from action_instruction"
                    )

        hover_will_be_performed = any(
            (s.get("action") or "").lower() in {
                "hover", "hover_element", "mouse_over"
            } for s in planned_steps
        )
        pre_hover_links = (
            await self.collect_visible_links()
            if hover_will_be_performed
            else []
        )

        for step in planned_steps:
            step_result = await self.run_step(step)
            self.logs.append(f"step: {step_result}")
            if (not step_result.get("ok")) and step.get("action") == "hover":
                # If planner hover target fails, retry with LLM-extracted target text.
                fallback_target = await self.extract_hover_target_from_instruction(
                    action_instruction
                )
                if fallback_target and fallback_target != (step.get("target") or ""):
                    self.logs.append(
                        f"hover_retry: llm extracted target='{fallback_target}'"
                    )
                    retry_step = {
                        "action": "hover",
                        "target": fallback_target,
                        "target_strategy": "text",
                        "timeout_ms": 8000,
                    }
                    retry_result = await self.run_step(retry_step)
                    self.logs.append(
                        f"step_retry_hover_from_instruction: {retry_result}"
                    )

        # If a hover was performed, capture a context snapshot of visible links NOW
        hover_context_links: List[Dict[str, str]] = []
        if hover_will_be_performed:
            post_hover_links = await self.collect_visible_links()
            pre_hover_keys = {
                f"{x.get('name', '')}|{x.get('url', '')}"
                for x in pre_hover_links
            }
            new_hover_links = [
                x for x in post_hover_links
                if f"{x.get('name', '')}|{x.get('url', '')}" not in pre_hover_keys
            ]
            hover_context_links = new_hover_links or post_hover_links
            self.logs.append(
                f"hover_context: captured {len(hover_context_links)} links "
                f"while dropdown active ({len(new_hover_links)} newly visible)"
            )

        total_timeout_sec = int(self.task.get("total_timeout_sec") or 300)
        max_pages = int(self.task.get("max_pages") or 1)
        max_repairs = int(self.task.get("max_repairs") or 2)
        required_fields = self.task.get("required_fields", [])
        all_rows: List[Dict[str, Any]] = []
        validation = {"ok": False, "reason": "Empty extracted data"}

        async def extraction_phase() -> Dict[str, Any]:
            """
            Main extraction phase of the dynamic scraping agent.
            It will try to extract the data from the page and if it fails, 
            it will try to repair itself and try again.
            """
            nonlocal validation
            for page_no in range(1, max_pages + 1):
                repairs_used = 0
                while repairs_used <= max_repairs:
                    # Use hover_context_links (captured while dropdown was active)
                    if hover_context_links:
                        source_rows = hover_context_links
                        self.logs.append(
                            f"hover_context: using pre-captured "
                            f"{len(source_rows)} links as source"
                        )
                    else:
                        links = await self.collect_visible_links()
                        source_rows = await self.build_source_rows(self.task, links)

                    source_keys = list(source_rows[0].keys()) if source_rows else []
                    field_map: Dict[str, str] = {}  # required_field → source_field
                    if source_keys and required_fields:
                        remap_prompt = f"""
Map each required field to the best matching source field name.

Required fields: {json.dumps(required_fields)}
Source fields:   {json.dumps(source_keys)}

Return ONLY JSON object: {{"required_field": "source_field", ...}}
Rules:
- Match by semantic meaning, not just exact name.
- Only include mappings where source_field != required_field (exact matches need no mapping).
- If no good match, omit the required field from the result.
- JSON only, no markdown."""
                        raw_map = await self.ask_ai(remap_prompt)
                        parsed_map = await self.safe_json_loads(raw_map, fallback={})
                        if isinstance(parsed_map, dict):
                            field_map = {
                                k: v for k, v in parsed_map.items()
                                if isinstance(k, str) and isinstance(v, str)
                                and k in required_fields and v in source_keys
                            }
                        self.logs.append(f"field_map: {field_map}")

                    def _remap_row(row: Dict) -> Dict:
                        """Rename source fields to required field names using LLM mapping."""
                        result = dict(row)
                        for req, src in field_map.items():
                            if req not in result or not result[req]:
                                if src in result and result[src]:
                                    result[req] = result[src]
                        return result

                    def _rows_with_required(rows, req_fields):
                        """
                        Filter rows that have all required fields.

                        Args:
                            rows (List[Dict[str, Any]]): List of rows to filter.
                            req_fields (List[str]): List of required fields.
                        
                        Returns:
                            List[Dict[str, Any]]: List of rows that have all required fields.
                        """
                        if not rows or not req_fields:
                            return []
                        return [
                            r for r in rows
                            if isinstance(r, dict) and all(
                                isinstance(r.get(f), str) and r.get(f, "").strip()
                                for f in req_fields
                            )
                        ]

                    def _rows_satisfy_required(rows, req_fields):
                        """
                        Check if rows satisfy required fields.

                        Args:
                            rows (List[Dict[str, Any]]): List of rows to check.
                            req_fields (List[str]): List of required fields.
                        
                        Returns:
                            bool: True if rows satisfy required fields, False otherwise.
                        """
                        return len(_rows_with_required(rows, req_fields)) >= 3

                    # Apply LLM field mapping to source rows.
                    remapped_rows = [_remap_row(r) for r in source_rows]

                    # Fast-path: skip map_to_schema when rows already have required fields.
                    use_fast_path = (
                        not hover_context_links
                        and _rows_satisfy_required(remapped_rows, required_fields)
                    )

                    if use_fast_path:
                        self.logs.append(
                            "map_to_schema: skipped — rows satisfy "
                            "required fields after remapping"
                        )
                        mapped = remapped_rows
                    else:
                        # Hover context rows are compact {name, url} pairs.
                        cap = 200 if hover_context_links else 100
                        
                        # Truncate extremely long string values to avoid hitting context limit
                        capped = []
                        for row in source_rows[:cap]:
                            trimmed = {}
                            for k, v in row.items():
                                if isinstance(v, str) and len(v) > 200:
                                    # Preserve URLs which might be long, but trim other text
                                    if k not in ("url", "href", "src") and not v.startswith("http"):
                                        trimmed[k] = v[:200] + "..."
                                    else:
                                        trimmed[k] = v
                                else:
                                    trimmed[k] = v
                            capped.append(trimmed)

                        llm_mapped = await self.map_to_schema(
                            instruction=self.task["extract_instruction"],
                            schema=self.task["output_schema"],
                            source_rows=capped,
                        )
                        if llm_mapped is not None:
                            mapped = llm_mapped
                            self.logs.append(
                                f"map_to_schema: llm returned "
                                f"{len(llm_mapped)} rows"
                            )
                            if hover_context_links and not mapped:
                                fallback_rows = _rows_with_required(
                                    remapped_rows, required_fields
                                )
                                if fallback_rows:
                                    mapped = fallback_rows
                                    self.logs.append(
                                        "map_to_schema: empty hover result, "
                                        f"falling back to {len(mapped)} "
                                        "captured dropdown rows"
                                    )
                        else:
                            # LLM returned invalid JSON — never discard extracted data.
                            mapped = remapped_rows
                            self.logs.append(
                                f"map_to_schema: llm invalid json, falling back to "
                                f"{len(remapped_rows)} remapped rows"
                            )

                    all_rows.extend(mapped)
                    dedup = {}
                    for row in all_rows:
                        dedup[json.dumps(row, sort_keys=True, ensure_ascii=False)] = row
                    data = list(dedup.values())
                    if required_fields:
                        data = [
                            row for row in data
                            if isinstance(row, dict) and all(
                                isinstance(row.get(f), str) and row.get(f, "").strip()
                                for f in required_fields
                            )
                        ]

                    if hasattr(self, 'last_extraction_spec') and self.last_extraction_spec:
                        skip_items = int(self.last_extraction_spec.get("skip_items") or 0)
                        max_items = int(self.last_extraction_spec.get("max_items") or 200)
                        if skip_items:
                            data = data[skip_items:]
                        if max_items:
                            data = data[:max_items]

                    validation = await self.validate_output(data, required_fields=required_fields)
                    if validation["ok"]:
                        return {
                            "ok": True,
                            "data": data,
                            "diagnostics": {
                                "logs": self.logs,
                                "executed_steps": self.executed_steps,
                            },
                        }

                    # Only run repair when source_rows itself is empty (page didn't
                    # load data). If we have rows but field validation fails, skip
                    # repair — it can't help and wastes 60-120s per cycle.
                    if source_rows:
                        self.logs.append(
                            f"repair: skipped — {len(source_rows)} source rows exist "
                            f"(validation failed: {validation['reason']})"
                        )
                        break

                    if repairs_used >= max_repairs:
                        break

                    observe = await self.observe_failure_context()
                    repair_steps = await self.propose_repair(
                        self.task, 
                        validation["reason"], 
                        observe
                    )
                    if not repair_steps:
                        self.logs.append("repair: no repair steps proposed")
                        break
                    self.logs.append(f"repair_steps: {repair_steps}")
                    for rstep in repair_steps[:3]:
                        rresult = await self.run_step(rstep)
                        self.logs.append(f"repair_step: {rresult}")
                    repairs_used += 1

                if page_no >= max_pages:
                    break

                # LLM-driven pagination: ask LLM to identify next-page mechanism
                # from visible links, handles any site (Next, ›, >, page numbers, etc.)
                page_links = await self.collect_visible_links()
                pagination_prompt = f"""
You are a pagination agent. Find the next-page navigation element.
Return ONLY JSON: {{"selector": "css_or_text", "strategy": "text|css|xpath", "found": true}}
Or {{"found": false}} if no next page exists.

Current page: {page_no} of {max_pages}
Visible links (sample):
{json.dumps(page_links[:60], ensure_ascii=False)}

Rules:
- Look for: "Next", "Next page", ">", "»", "›", next page number (e.g. current=1 then look for "2").
- If found, set selector to the link text or CSS/xpath selector.
- No markdown. JSON only."""
                pagination_raw = await self.ask_ai(pagination_prompt)
                pagination_parsed = await self.safe_json_loads(
                    pagination_raw, 
                    fallback={"found": False}
                )
                if not (
                    isinstance(pagination_parsed, dict) and 
                    pagination_parsed.get("found")
                ):
                    self.logs.append("pagination: next button not found (llm)")
                    break

                nav_selector = pagination_parsed.get("selector", "")
                nav_strategy = pagination_parsed.get("strategy", "text")
                nav_step = {
                    "action": "click",
                    "target": nav_selector,
                    "target_strategy": nav_strategy,
                    "timeout_ms": 5000,
                }
                nav_result = await self.run_step(nav_step)
                self.logs.append(f"pagination: {nav_result}")
                if not nav_result.get("ok"):
                    self.logs.append("pagination: click failed, stopping")
                    break
                
                await self.page.wait_for_load_state("domcontentloaded")
                await self.page.wait_for_timeout(2000)

            dedup = {}
            for row in all_rows:
                dedup[json.dumps(row, sort_keys=True, ensure_ascii=False)] = row
            
            data = list(dedup.values())
            if required_fields:
                data = [
                    row for row in data
                    if isinstance(row, dict) and all(
                        isinstance(row.get(f), str) and row.get(f, "").strip()
                        for f in required_fields
                    )
                ]
            
            observe = await self.observe_failure_context()
            return {
                "ok": False,
                "data": data,
                "diagnostics": {
                    "reason": validation["reason"],
                    "logs": self.logs,
                    "executed_steps": self.executed_steps,
                    "observe": observe,
                },
            }

        try:
            return await asyncio.wait_for(
                extraction_phase(), 
                timeout=total_timeout_sec
            )
        except asyncio.TimeoutError:
            self.logs.append(
                f"timeout: total task timeout reached ({total_timeout_sec}s)"
            )
            
            dedup = {}
            for row in all_rows:
                dedup[json.dumps(row, sort_keys=True, ensure_ascii=False)] = row
            
            data = list(dedup.values())
            if required_fields:
                data = [
                    row for row in data
                    if isinstance(row, dict) and all(
                        isinstance(row.get(f), str) and row.get(f, "").strip()
                        for f in required_fields
                    )
                ]

            if hasattr(self, 'last_extraction_spec') and self.last_extraction_spec:
                skip_items = int(self.last_extraction_spec.get("skip_items") or 0)
                max_items = int(self.last_extraction_spec.get("max_items") or 200)
                if skip_items:
                    data = data[skip_items:]
                if max_items:
                    data = data[:max_items]

            observe = await self.observe_failure_context()
            return {
                "ok": False,
                "data": data,
                "diagnostics": {
                    "reason": f"Task timed out after {total_timeout_sec} seconds.",
                    "logs": self.logs,
                    "executed_steps": self.executed_steps,
                    "observe": observe,
                },
            }
