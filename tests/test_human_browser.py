"""Frozen-observation DOM inspection for human demonstrations; no browser needed."""
import json
import shutil
import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from osworld_agent.browser_session import BrowserOptions, BrowserSession


class InspectCDP:
    def __init__(self):
        self.document_id = 10
        self.frame_id = "main"
        self.hit_frame = "main"
        self.hit = "span"
        self.focused = "button"
        self.calls = []
        self.threads = []
        self.ancestors = {"span": "button", "button": None}
        self.backend_ids = {"span": 2, "button": 3, "canvas": 4, "input": 5, "iframe": 6}
        self.properties = {
            "span": {"tag": "span", "editable": False, "sensitive": False, "sensitive_reasons": [], "bounds": [5, 6, 20, 10]},
            "button": {"tag": "button", "editable": False, "sensitive": False, "sensitive_reasons": [], "bounds": [0, 0, 50, 30]},
            "canvas": {"tag": "canvas", "editable": False, "sensitive": False, "sensitive_reasons": [], "bounds": [0, 0, 800, 600]},
            "input": {"tag": "input", "editable": True, "sensitive": True, "sensitive_reasons": ["password"], "bounds": [0, 0, 100, 30]},
            "iframe": {"tag": "iframe", "editable": False, "sensitive": False, "sensitive_reasons": [], "bounds": [0, 0, 300, 200]},
        }
        self.navigate_during_inspection = False

    def detach(self):
        pass

    def send(self, command, params=None):
        self.calls.append((command, params))
        self.threads.append(threading.get_ident())
        if command == "Target.getTargetInfo":
            return {"targetInfo": {"targetId": "page"}}
        if command == "DOM.getDocument":
            return {"root": {"backendNodeId": self.document_id}}
        if command == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": self.frame_id}}}
        if command == "DOM.getNodeForLocation":
            return {"backendNodeId": self.backend_ids[self.hit], "frameId": self.hit_frame}
        if command == "DOM.resolveNode":
            obj = next(key for key, value in self.backend_ids.items() if value == params["backendNodeId"])
            return {"object": {"objectId": obj}}
        if command == "Runtime.evaluate":
            return {"result": {"objectId": self.focused}} if self.focused else {"result": {"subtype": "null"}}
        if command == "DOM.describeNode":
            return {"node": {"backendNodeId": self.backend_ids[params["objectId"]]}}
        if command == "Runtime.callFunctionOn":
            obj = params["objectId"]
            if params["returnByValue"]:
                if "same_focus" in params["functionDeclaration"]:
                    return {"result": {"value": {"same_focus": obj == self.focused, "fields": dict(self.properties[obj])}}}
                if self.navigate_during_inspection:
                    self.document_id += 1
                return {"result": {"value": dict(self.properties[obj])}}
            skip_self = params["arguments"][0]["value"]
            candidate = self.ancestors.get(obj) if skip_self or obj == "span" else obj
            return {"result": {"objectId": candidate}} if candidate else {"result": {"subtype": "null"}}
        return {}


class HumanBrowserInspectionTests(unittest.TestCase):
    def setUp(self):
        self.cdp = InspectCDP()
        self.viewport = {"w": 800, "h": 600, "x": 0, "y": 100}
        self.page = SimpleNamespace(url="https://example.test", evaluate=lambda _: dict(self.viewport),
                                    context=SimpleNamespace(new_cdp_session=lambda _: self.cdp),
                                    mouse=Mock(), keyboard=Mock())
        self.browser = SimpleNamespace(close=Mock())
        self.session = BrowserSession(BrowserOptions(require_focus=False))
        self.session._connect = Mock(return_value=(SimpleNamespace(stop=Mock()), self.browser))
        self.session._pick_page = Mock(return_value=self.page)
        self.metadata = {"page_id": "page", "document_id": 10, "snapshot_id": "snapshot", "captured_at": time.time(),
                         "url": self.page.url, "scroll_offset": [0, 100], "screenshot_space": "css_viewport",
                         "screenshot_size": [800, 600], "exposed_refs": ["7"]}
        self.session._observation_metadata = dict(self.metadata)
        self.session.snapshot = {**self.metadata, "frame_id": "main"}
        self.session.nodes = {"7": {"ref": "7", "backend_id": 3, "direct": True, "frame_id": "main",
                                    "role": "button", "name": "Submit", "id": "submit"}}
        self.addCleanup(self.session.close)

    def test_live_child_hit_maps_exact_interactive_ancestor_without_refresh(self):
        old_snapshot = self.session.snapshot
        result = self.session.inspect_point(15.5, 20.5, self.metadata)
        self.assertEqual(result["status"], "inspected")
        self.assertTrue(result["matched"])
        self.assertEqual(result["target_ref"], "7")
        self.assertEqual(result["target_label"], "button Submit")
        self.assertEqual(result["element_id"], "submit")
        self.assertEqual(result["bounds"], [0, 0, 50, 30])
        self.assertEqual(result["point"], [15.5, 20.5])
        self.assertIs(self.session.snapshot, old_snapshot)
        self.assertFalse(result["dispatched"])
        self.assertEqual(len(set(self.cdp.threads)), 1)
        self.assertNotEqual(self.cdp.threads[0], threading.get_ident())
        self.assertEqual(self.page.mouse.mock_calls, [])
        self.assertEqual(self.page.keyboard.mock_calls, [])
        self.assertTrue(any(command == "Runtime.releaseObjectGroup" for command, _ in self.cdp.calls))
        self.assertFalse(any(command == "DOMSnapshot.captureSnapshot" for command, _ in self.cdp.calls))

    def test_stale_metadata_rejected_before_dom_hit_test(self):
        metadata = {**self.metadata, "snapshot_id": "old"}
        result = self.session.inspect_point(10, 20, metadata)
        self.assertEqual(result["error_code"], "stale")
        self.assertFalse(result["dispatched"])
        self.session._connect.assert_not_called()

    def test_same_url_document_change_rejected(self):
        self.cdp.document_id = 11
        result = self.session.inspect_focused(self.metadata)
        self.assertEqual(result["error_code"], "stale")
        self.assertFalse(any(command == "Runtime.evaluate" for command, _ in self.cdp.calls))

    def test_scroll_change_rejected_before_hit_test(self):
        self.viewport["y"] = 150
        result = self.session.inspect_point(10, 20, self.metadata)
        self.assertEqual(result["error_code"], "stale")
        self.assertFalse(any(command == "DOM.getNodeForLocation" for command, _ in self.cdp.calls))

    def test_document_change_during_inspection_rejected(self):
        self.cdp.navigate_during_inspection = True
        result = self.session.inspect_point(10, 20, self.metadata)
        self.assertEqual(result["error_code"], "stale")
        self.assertNotIn("target_ref", result)

    def test_unexposed_node_keeps_real_label_but_omits_reference(self):
        metadata = {**self.metadata, "exposed_refs": []}
        result = self.session.inspect_point(10, 20, metadata)
        self.assertTrue(result["matched"])
        self.assertTrue(result["unexposed"])
        self.assertNotIn("target_ref", result)
        self.assertNotIn("element_id", result)
        self.assertEqual(result["target_label"], "button Submit")

    def test_unknown_backend_is_not_guessed_from_bbox(self):
        self.session.nodes["7"]["backend_id"] = 99
        result = self.session.inspect_point(10, 20, self.metadata)
        self.assertFalse(result["matched"])
        self.assertNotIn("target_ref", result)
        self.assertEqual(result["target_label"], "")

    def test_canvas_remains_unmatched(self):
        self.cdp.hit = "canvas"
        result = self.session.inspect_point(10, 20, self.metadata)
        self.assertEqual(result["reason"], "canvas_target")
        self.assertFalse(result["matched"])
        self.assertEqual(result["target_label"], "")

    def test_child_frame_does_not_invent_main_frame_target(self):
        self.cdp.hit_frame = "child"
        result = self.session.inspect_point(10, 20, self.metadata)
        self.assertEqual(result["reason"], "iframe_target")
        self.assertTrue(result["sensitive_unknown"])
        self.assertFalse(result["editable"])
        self.assertNotIn("target_ref", result)

    def test_focused_password_marks_sensitive_without_value(self):
        self.cdp.focused = "input"
        self.cdp.ancestors["input"] = None
        self.session.nodes["7"].update(backend_id=5, role="textbox", name="Password")
        result = self.session.inspect_focused(self.metadata)
        self.assertEqual(result["target_ref"], "7")
        self.assertTrue(result["editable"])
        self.assertTrue(result["sensitive"])
        self.assertIn("password", result["sensitive_reasons"])
        self.assertNotIn("value", result)
        script = next(params["expression"] for command, params in self.cdp.calls if command == "Runtime.evaluate")
        self.assertIn("shadowRoot.activeElement", script)
        self.assertNotIn(".value", script)

    def test_focused_iframe_does_not_claim_editable_parent(self):
        self.cdp.focused = "iframe"
        result = self.session.inspect_focused(self.metadata)
        self.assertTrue(result["sensitive_unknown"])
        self.assertFalse(result["editable"])
        self.assertNotIn("target_ref", result)

    def test_out_of_bounds_does_not_hit_test(self):
        result = self.session.inspect_point(800, 20, self.metadata)
        self.assertEqual(result["error_code"], "not_found")
        self.assertFalse(any(command == "DOM.getNodeForLocation" for command, _ in self.cdp.calls))

    def test_pointer_scroll_rejects_partial_or_invalid_coordinates(self):
        for coordinates in ({"x": 10}, {"x": True, "y": 20}, {"x": 800, "y": 20},
                            {"x": 20, "y": float("nan")}):
            with self.subTest(coordinates=coordinates):
                action = SimpleNamespace(action="scroll", args={"direction": "down", "amount": 1, **coordinates})
                result = self.session.execute_viewport_action(action, self.metadata)
                self.assertEqual(result["status"], "rejected")
                self.assertFalse(result["dispatched"])
        self.assertEqual(self.page.mouse.mock_calls, [])
        self.session._connect.assert_not_called()

    def test_pointer_scroll_validates_offset_before_input(self):
        self.viewport["y"] = 150
        action = SimpleNamespace(action="scroll", args={"direction": "down", "amount": 1, "x": 20, "y": 30})
        result = self.session.execute_viewport_action(action, self.metadata)
        self.assertEqual(result["error_code"], "stale")
        self.assertEqual(self.page.mouse.mock_calls, [])

    def prepare_editable_input(self, sensitive=False):
        self.cdp.focused = "input"
        self.cdp.properties["input"].update(sensitive=sensitive, sensitive_reasons=["password"] if sensitive else [])
        self.session.nodes["7"].update(backend_id=5, role="textbox", name="Input")

    def test_focused_text_uses_keyboard_without_mouse_or_focus_reset(self):
        self.prepare_editable_input()
        result = self.session.insert_focused_text("中文", self.metadata, expected_backend_id=5)
        self.assertEqual(result["status"], "executed")
        self.page.keyboard.insert_text.assert_called_once_with("中文")
        self.page.keyboard.press.assert_not_called()
        self.assertEqual(self.page.mouse.mock_calls, [])

    def test_focused_text_rejects_sensitive_input(self):
        self.prepare_editable_input(sensitive=True)
        result = self.session.insert_focused_text("not dispatched", self.metadata, expected_backend_id=5)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "blocked")
        self.assertFalse(result["dispatched"])
        self.assertEqual(self.page.keyboard.mock_calls, [])

    def test_focused_text_rejects_expected_element_mismatch(self):
        self.prepare_editable_input()
        result = self.session.insert_focused_text("not dispatched", self.metadata, expected_backend_id=99)
        self.assertEqual(result["error_code"], "stale")
        self.assertEqual(self.page.keyboard.mock_calls, [])

    def test_focused_text_rechecks_focus_immediately_before_dispatch(self):
        self.prepare_editable_input()
        picks = []
        def pick_page(browser):
            picks.append(1)
            if len(picks) == 2:
                self.cdp.focused = "button"
            return self.page
        self.session._pick_page.side_effect = pick_page
        result = self.session.insert_focused_text("not dispatched", self.metadata, expected_backend_id=5)
        self.assertEqual(result["error_code"], "stale")
        self.assertEqual(self.page.keyboard.mock_calls, [])

    def test_focused_text_transport_error_after_dispatch_is_uncertain(self):
        self.prepare_editable_input()
        self.page.keyboard.insert_text.side_effect = RuntimeError("transport lost")
        result = self.session.insert_focused_text("中文", self.metadata, expected_backend_id=5)
        self.assertEqual(result["status"], "uncertain")
        self.assertTrue(result["dispatched"])
        self.assertEqual(self.page.mouse.mock_calls, [])

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute the DOM-property inspector without a browser")
    def test_sensitive_field_javascript_returns_only_text_presence(self):
        script = """
          const fs = require('fs');
          const {fn, cases} = JSON.parse(fs.readFileSync(0, 'utf8'));
          const inspect = eval('(' + fn + ')');
          const results = cases.map(attrs => {
            const element = {nodeType:1,isConnected:true,tagName:'INPUT',id:attrs.id || '',
              labels:[],disabled:false,readOnly:false,isContentEditable:false,
              getAttribute:key => attrs[key] || null, closest:() => null,
              getBoundingClientRect:() => ({x:0,y:0,width:100,height:20})};
            Object.defineProperty(element, 'value', {get(){return 'private-input-content-fixture';}});
            return inspect.call(element);
          });
          process.stdout.write(JSON.stringify(results));
        """
        cases = [{"type": "password"}, {"autocomplete": "section-login one-time-code"},
                 {"autocomplete": "cc-number"}, {"name": "cardNumber"},
                 {"aria-label": "验证码"}, {"id": "otp_code"}, {"aria-label": "Verification code"},
                 {"name": "cc_number"}, {"type": "text", "name": "search"}]
        completed = subprocess.run([shutil.which("node"), "-e", script],
            input=json.dumps({"fn": BrowserSession._INSPECT_PROPERTIES, "cases": cases}),
            text=True, capture_output=True, timeout=10, check=True)
        results = json.loads(completed.stdout)
        self.assertEqual([item["sensitive"] for item in results], [True] * 8 + [False])
        self.assertTrue(all("value" not in item for item in results))
        self.assertTrue(all(item["editable_has_text"] is True for item in results))
        self.assertNotIn("private-input-content-fixture", completed.stdout)


if __name__ == "__main__":
    unittest.main()
