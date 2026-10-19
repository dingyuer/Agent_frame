"""Regression for enterprise card grids that have text AX but no card refs."""
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image, ImageDraw

from osworld_agent.actions import Action
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.env import Observation
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.scene import browser_context, browser_scene


def screenshot():
    image = Image.new('RGB', (1200, 800), 'white')
    draw = ImageDraw.Draw(image)
    for i in range(30):
        x, y = (i % 3) * 390 + 20, (i // 3) * 72 + 20
        draw.rectangle((x, y, x + 350, y + 60), fill='#71a6cb')
    return image


def card_page():
    nodes = [{'ref': str(i), 'role': 'StaticText', 'name': f'Solution card {i}', 'direct': False}
             for i in range(348)]
    nodes += [{'ref': str(400 + i), 'role': 'image', 'name': 'thumbnail'} for i in range(59)]
    nodes += [{'ref': str(500 + i), 'role': 'link', 'name': 'Solution Library', 'direct': True}
              for i in range(6)]
    nodes += [{'ref': str(600 + i), 'role': 'button', 'name': 'Filter', 'direct': True}
              for i in range(10)]
    nodes += [{'ref': str(700 + i), 'role': 'treeitem', 'name': 'category', 'direct': True}
              for i in range(40)]
    return {'kind': 'browser_ax', 'snapshot_id': 'snap-1', 'url': 'https://intranet.test/#/library', 'nodes': nodes}


def test_content_cards_need_visual_targeting_but_ax_stays_available():
    page, image = card_page(), screenshot()
    scene = browser_scene(page, image)
    assert scene.mode == 'browser_gui'
    assert scene.structure_available
    assert scene.reason == 'content_without_actionable_ax'
    obs = Observation(screenshot=image, context=[page], info={'scene': scene.to_dict()})
    assert browser_context(obs) == [page]


def test_full_observation_and_route_do_not_map_card_to_navigation():
    page, image = card_page(), screenshot()
    session = Mock()
    session.collect_observation.return_value = (image, page, {
        'screenshot_space': 'css_viewport', 'page_id': 'page-1', 'snapshot_id': 'snap-1'})
    session.route.return_value = None
    env = WindowsEnvironment()
    env._get_browser_session = Mock(return_value=session)
    env._runtime_state = Mock(return_value={})
    obs = env._build_observation()
    assert obs.info['scene']['mode'] == 'browser_gui'
    action = Action('click', {'target': 'UDG DC05 Cutover Plan'})
    assert env.route_action(action, obs) is None
    session.resolve_target.assert_not_called()
    # An explicit AX reference remains usable in mixed pages.
    session.route.return_value = {'channel': 'browser_dom', 'target_ref': '500'}
    assert env.route_action(Action('click', {'target': 'Solution Library', 'target_ref': '500'}), obs)['channel'] == 'browser_dom'


def test_ax_ref_repair_does_not_block_visual_card_click():
    page, image = card_page(), screenshot()
    obs = Observation(screenshot=image, context=[page], info={
        'scene': browser_scene(page, image).to_dict(), 'exposed_refs': ['500', '600']})
    pipe = Pipeline.__new__(Pipeline)
    assert pipe._dom_reference_feedback(Action('click', {'target': 'UDG DC05 Cutover Plan'}), StepState(obs=obs)) is None


def test_actionable_results_and_blank_page_do_not_force_gui():
    page = card_page()
    page['nodes'] += [{'ref': str(800 + i), 'role': 'link', 'name': f'Paper {i}', 'direct': True}
                      for i in range(100)]
    assert browser_scene(page, screenshot()).mode == 'browser_dom'
    assert browser_scene(card_page(), Image.new('RGB', (1200, 800), 'black')).mode == 'browser_dom'
