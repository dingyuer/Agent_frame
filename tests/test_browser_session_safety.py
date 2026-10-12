"""Browser identity, input dispatch and downloads; no live browser or model required."""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from osworld_agent.browser_session import BrowserOptions, BrowserSession


def action(name="click", **args):
    return SimpleNamespace(action=name, args=args)


class FakeCDP:
    def __init__(self, page):
        self.page = page

    def detach(self):
        pass

    def send(self, command, params=None):
        page = self.page
        if command == "Target.getTargetInfo":
            return {"targetInfo": {"targetId": page.page_id, "openerId": page.opener_id}}
        if command == "DOM.getDocument":
            return {"root": {"backendNodeId": page.document_id}}
        if command == "DOMSnapshot.captureSnapshot":
            if page.ax_failed:
                raise RuntimeError("AX unavailable")
            return {"strings": ["BUTTON", "frame"], "documents": [{
                "frameId": 1,
                "nodes": {"backendNodeId": [42], "nodeName": [0], "attributes": [[]]},
                "layout": {"nodeIndex": [0], "bounds": [[10, 20, 40, 30]]},
            }]}
        if command == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "frame"}}}
        if command == "Accessibility.getFullAXTree":
            return {"nodes": [{"nodeId": "ax1", "backendDOMNodeId": 42,
                               "role": {"value": "button"}, "name": {"value": "Submit"}}]}
        if command == "DOM.resolveNode":
            return {"object": {"objectId": "element"}}
        if command == "Runtime.callFunctionOn":
            if "getBoundingClientRect" in params["functionDeclaration"]:
                value = {"x": 30, "y": 35, "w": 40, "h": 30, "tag": "button",
                         "readOnly": False, "editable": False}
            else:
                value = True
            return {"result": {"value": value}}
        return {}


class FakePage:
    def __init__(self, page_id="page1", opener_id=None):
        self.page_id = page_id
        self.opener_id = opener_id
        self.document_id = 11
        self.url = "https://example.test/form"
        self.frames = [object()]
        self.mouse = Mock()
        self.keyboard = Mock()
        self.focused = True
        self.ax_failed = False
        self.scroll_y = 0
        self.context = SimpleNamespace(new_cdp_session=lambda _: FakeCDP(self))
        self.screenshot = Mock(return_value=b"fake PNG")

    def evaluate(self, script, *args):
        if "cloneNode" in script:
            return self.content()
        if "document.hasFocus" in script:
            return self.focused
        if "width:innerWidth" in script:
            return {"width": 800, "height": 600, "x": 0, "y": self.scroll_y, "dpr": 2}
        if "w: innerWidth" in script:
            return {"w": 800, "h": 600, "x": 0, "y": self.scroll_y}
        return None

    def title(self):
        return "Form"

    def content(self):
        return f"<main>{self.page_id}</main>"

    def wait_for_timeout(self, timeout):
        pass


class BrowserSessionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.page = FakePage()
        self.context = SimpleNamespace(pages=[self.page])
        self.browser = SimpleNamespace(contexts=[self.context], close=Mock())
        self.session = BrowserSession(BrowserOptions(require_focus=False))
        self.session._connect = Mock(return_value=(SimpleNamespace(stop=Mock()), self.browser))
        self.image = SimpleNamespace(size=(800, 600), load=Mock())
        fake_pil = SimpleNamespace(Image=SimpleNamespace(open=Mock(return_value=self.image)))
        self.pil_patch = patch.dict(sys.modules, {"PIL": fake_pil})
        self.pil_patch.start()
        self.addCleanup(self.pil_patch.stop)
        self.addCleanup(self.session.close)

    def observe(self):
        return self.session.collect_observation()

    def route(self):
        return {"target_ref": "0", "snapshot_id": self.session.snapshot["snapshot_id"]}

    def test_observation_uses_one_connection_and_css_pixels(self):
        shot, snapshot, metadata = self.observe()
        self.assertIs(shot, self.image)
        self.session._connect.assert_called_once()
        self.page.screenshot.assert_called_once_with(full_page=False, scale="css")
        self.assertEqual(metadata["screenshot_size"], [800, 600])
        self.assertEqual(metadata["screenshot_space"], "css_viewport")
        self.assertEqual(snapshot["snapshot_id"], metadata["snapshot_id"])
        self.assertEqual((metadata["page_id"], metadata["document_id"]), ("page1", 11))

    def test_failed_connection_clears_old_snapshot(self):
        self.observe()
        self.session._connect.side_effect = RuntimeError("offline")
        self.assertIsNone(self.session.collect())
        self.assertIsNone(self.session.snapshot)
        self.assertEqual(self.session.nodes, {})

    def test_missing_page_clears_old_snapshot(self):
        self.observe()
        self.context.pages = []
        self.assertIsNone(self.session.collect())
        self.assertEqual(self.session.nodes, {})

    def test_ax_failure_keeps_fresh_bound_screenshot(self):
        self.observe()
        self.page.ax_failed = True
        shot, snapshot, metadata = self.observe()
        self.assertIs(shot, self.image)
        self.assertIsNone(snapshot)
        self.assertEqual(self.session.nodes, {})
        self.assertEqual(metadata["screenshot_space"], "css_viewport")
        self.assertTrue(metadata["snapshot_id"])
        result = self.session.execute_viewport_action(action(x=20, y=30), metadata)
        self.assertEqual(result["status"], "executed")

    def test_screenshot_failure_does_not_label_desktop_as_viewport(self):
        self.page.screenshot.side_effect = RuntimeError("capture failed")
        shot, snapshot, metadata = self.observe()
        self.assertIsNone(shot)
        self.assertIsNotNone(snapshot)
        self.assertNotIn("screenshot_space", metadata)

    def test_navigation_between_ax_and_screenshot_discards_both(self):
        def navigate(**kwargs):
            self.page.document_id += 1
            return b"fake PNG"
        self.page.screenshot.side_effect = navigate
        shot, snapshot, metadata = self.observe()
        self.assertIsNone(shot)
        self.assertIsNone(snapshot)
        self.assertEqual(metadata["error_code"], "stale")
        self.assertEqual(self.session.nodes, {})

    def test_stale_route_does_not_connect_or_dispatch(self):
        self.observe()
        self.session._connect.reset_mock()
        result = self.session.execute(action(target_ref="0"), {"target_ref": "0", "snapshot_id": "old"})
        self.assertEqual(result["error_code"], "stale")
        self.assertFalse(result["dispatched"])
        self.session._connect.assert_not_called()
        self.page.mouse.click.assert_not_called()

    def test_expired_snapshot_is_rejected(self):
        self.observe()
        self.session.snapshot["captured_at"] = time.time() - 1000
        result = self.session.execute(action(), self.route())
        self.assertEqual(result["error_code"], "stale")
        self.page.mouse.click.assert_not_called()

    def test_same_url_document_reload_rejects_old_node(self):
        self.observe()
        self.page.document_id += 1
        result = self.session.execute(action(), self.route())
        self.assertEqual(result["error_code"], "stale")
        self.page.mouse.click.assert_not_called()

    def test_tab_binding_ignores_popups_until_explicit_switch_and_returns_to_parent(self):
        unrelated = FakePage("unrelated")
        self.context.pages.append(unrelated)
        self.assertIs(self.session._pick_page(self.browser), self.page)
        self.context.pages.reverse()
        self.assertIs(self.session._pick_page(self.browser), self.page)
        popup = FakePage("popup", opener_id="page1")
        self.context.pages.append(popup)
        self.assertIs(self.session._pick_page(self.browser), self.page)
        self.assertEqual(self.session.switch_tab("popup")["status"], "executed")
        self.assertIs(self.session._pick_page(self.browser), popup)
        self.context.pages.remove(popup)
        self.assertIs(self.session._pick_page(self.browser), self.page)

    def test_reconnected_page_wrapper_uses_protocol_identity(self):
        self.observe()
        replacement = FakePage("page1")
        self.context.pages = [FakePage("other"), replacement]
        result = self.session.execute(action(), self.route())
        self.assertEqual(result["status"], "executed")
        replacement.mouse.click.assert_called_once()
        self.page.mouse.click.assert_not_called()

    def test_source_uses_bound_page_after_tab_reordering(self):
        self.observe()
        self.context.pages.insert(0, FakePage("other"))
        source = self.session.get_page_source()
        self.assertIn("<main>page1</main>", source)
        self.assertNotIn("<main>other</main>", source)

    def test_pdf_link_click_retains_click_semantics(self):
        self.observe()
        self.session.nodes["0"]["href"] = "https://example.test/report.pdf"
        self.session.download = Mock()
        result = self.session.execute(action(), self.route())
        self.assertEqual(result["status"], "executed")
        self.page.mouse.click.assert_called_once()
        self.session.download.assert_not_called()

    def test_click_timeout_after_dispatch_is_uncertain(self):
        self.observe()
        self.page.mouse.click.side_effect = RuntimeError("transport lost")
        result = self.session.execute(action(), self.route())
        self.assertEqual(result["error_code"], "uncertain")
        self.assertTrue(result["dispatched"])

    def test_visual_rejects_stale_document(self):
        _, _, metadata = self.observe()
        self.page.document_id += 1
        result = self.session.execute_viewport_action(action(x=20, y=30), metadata)
        self.assertEqual(result["error_code"], "stale")
        self.page.mouse.click.assert_not_called()

    def test_visual_rejects_scroll_changed_after_screenshot(self):
        _, _, metadata = self.observe()
        self.page.scroll_y = 200
        result = self.session.execute_viewport_action(action(x=20, y=30), metadata)
        self.assertEqual(result["error_code"], "stale")
        self.page.mouse.click.assert_not_called()

    def test_cross_frame_scroll_does_not_use_iframe_coordinates(self):
        _, snapshot, metadata = self.observe()
        self.session.nodes["0"].update(frame_id="child-frame", direct=False)
        obs = SimpleNamespace(info={**metadata, "exposed_refs": ["0"]})
        route = self.session.route(action("scroll", target_ref="0"), obs)
        self.assertEqual(route["error_code"], "unsupported")
        result = self.session.execute(action("scroll", amount=1, direction="down"), self.route())
        self.assertEqual(result["error_code"], "unsupported")
        self.page.mouse.wheel.assert_not_called()

    def test_visual_rejects_out_of_bounds_and_unknown_select(self):
        _, _, metadata = self.observe()
        result = self.session.execute_viewport_action(action(x=800, y=30), metadata)
        self.assertFalse(result["dispatched"])
        result = self.session.execute_viewport_action(action("select", x=20, y=30, option="A"), metadata)
        self.assertEqual(result["error_code"], "unsupported")
        self.page.mouse.click.assert_not_called()

    def test_visual_drag_uses_cdp_and_releases_button_on_error(self):
        _, _, metadata = self.observe()
        self.page.mouse.down.side_effect = RuntimeError("transport lost")
        result = self.session.execute_viewport_action(action("drag_and_drop", x1=20, y1=30, x2=200, y2=300), metadata)
        self.assertEqual(result["error_code"], "uncertain")
        self.assertTrue(result["dispatched"])
        self.page.mouse.up.assert_called_once()

    def test_keyboard_action_needs_no_coordinates(self):
        _, _, metadata = self.observe()
        result = self.session.execute_viewport_action(action("hotkey", keys=["ctrl", "a"]), metadata)
        self.assertEqual(result["status"], "executed")
        self.page.keyboard.press.assert_called_once_with("ControlOrMeta+a")

    def test_scroll_action_needs_no_coordinates(self):
        _, _, metadata = self.observe()
        result = self.session.execute_viewport_action(action("scroll", amount=1, direction="down"), metadata)
        self.assertEqual(result["status"], "executed")

    def test_direct_download_never_requests_a_resource(self):
        url = "https://example.test/report.pdf?token=" + "a" * 250
        get = Mock()
        self.page.context.request = SimpleNamespace(get=get)
        result = self.session.download(url)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["dispatched"])
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
