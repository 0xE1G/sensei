"""Two-minute check that the core works. Run after every change.

    python scripts/smoke_test.py

Builds each task scene, renders the cameras, checks depth-based perception
against true geometry, moves the arm, runs one block pick-and-place, and checks
the grasp sampler's geometry on a bowl and a block.
Writes images to runs/smoke/.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from kintsugi.agents.jobs.task_spec import to_task  # noqa: E402
from kintsugi.agents.llm import ModelClient  # noqa: E402
from kintsugi.executor import run_episode  # noqa: E402
from kintsugi.investigate import Investigator  # noqa: E402
from kintsugi.oracle import run_oracle  # noqa: E402
from kintsugi.composites import applicable, signature  # noqa: E402
from kintsugi.primitive_code import check_primitive  # noqa: E402
from kintsugi.program import check_program  # noqa: E402
from kintsugi.skills.library import seed_library  # noqa: E402
from kintsugi.tasks.specs import probe  # noqa: E402
from kintsugi.perception.camera import Camera  # noqa: E402
from kintsugi.perception.perceive import Perception  # noqa: E402
from kintsugi.search.failure_search import SPACE, XY, plausible  # noqa: E402
from kintsugi.search.grasp_sampler import family_of, geometric_fit, sample_candidates  # noqa: E402
from kintsugi.skills.library import LIB_DIR, load_card_file  # noqa: E402
from kintsugi.sim.robot import Robot, down_rotation  # noqa: E402
from kintsugi.sim.scene import build_scene  # noqa: E402
from kintsugi.tasks.specs import TASKS, make_task  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "smoke"


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    import imageio.v2 as imageio
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for name in TASKS:
        task = make_task(name, seed=1)
        scene = build_scene(task, seed=1)
        cam = Camera(scene.model, scene.data, "front", 640, 480)
        imageio.imwrite(OUT / f"{name}_front.png", cam.rgb())
        d = cam.depth()
        imageio.imwrite(OUT / f"{name}_depth.png", (255 * np.clip(d / d.max(), 0, 1)).astype(np.uint8))
        cam.close()
        perc_sim, perc_gt = Perception(scene, "sim"), Perception(scene, "ground_truth")
        ps, pg = perc_sim.perceive(), perc_gt.perceive()
        # an object whose moving part starts open (a swung-out door) really extends beyond its body origin: not a
        # perception error, so its centre is not compared
        opened = {o["id"] for o in task["objects"] if (o.get("params") or {}).get("open")}
        errs = [np.linalg.norm(np.array(ps[o]["center"]) - pg[o]["center"]) for o in ps if ps[o] and o not in opened] or [0.0]
        results.append(check(f"scene+perception {name}", all(ps.values()) and max(errs) < 0.015,
                             f"objects={len(ps)} max_centre_err={1000 * max(errs):.1f}mm"))
        perc_sim.close()

    scene = build_scene(make_task("single_block", seed=0), seed=0)
    robot = Robot(scene)
    r = robot.move_to([0.5, 0.15, 0.2], down_rotation(0.3))
    results.append(check("arm reaches a pose", r["reached"], f"err={1000 * r['pos_err']:.1f}mm"))

    rec, _ = run_episode(make_task("single_block", seed=0), seed=0)
    st = rec["subtasks"][0]
    results.append(check("pick and place a block", rec["success"],
                         f"label={st['label']} trace={[(s['skill'], s['ok']) for s in st['trace']]}"))
    # Grasp sampler (no physics): on a bowl only rim grasps fit the gripper; on a block, centre grasps do.
    prim = load_card_file(LIB_DIR / "primitives" / "feature_pinch.json")
    cands = sample_candidates(prim, 400, np.random.default_rng(0))
    fams = {}
    for task, obj in (("bowl_on_tray", "bowl"), ("single_block", "block")):
        sc = build_scene(make_task(task, seed=1), seed=1)
        per = Perception(sc)
        d = per.perceive([obj])[obj]
        fams[obj] = {family_of(d, p)["anchor"] for p in cands if geometric_fit(per.last_points[obj], d, p)["ok"]}
        per.close()
    results.append(check("grasp sampler", fams["bowl"] == {"rim"} and "centre" in fams["block"], str(fams)))
    # Oracle: privileged expert solves the block scene.
    o = run_oracle(make_task("single_block", seed=0), seed=0)
    results.append(check("oracle solves a block scene", o["success"], str(o["subtask_ok"])))

    # Model layer on the rule backend: ok / refuse / clarify, all schema-valid; the ok spec builds a scene.
    client = ModelClient("rule", use_cache=False)
    outs = {p: client.complete("task_spec", {"prompt": p}).output for p in
            ("Put the bowl on the tray.", "Pour water into the cup.", "Tidy up please.")}
    statuses = [v["status"] for v in outs.values()]
    spec = to_task(outs["Put the bowl on the tray."], {"prompt": "Put the bowl on the tray."})
    build_scene(spec, seed=3)
    results.append(check("model layer (rule backend)", statuses == ["ok", "refuse", "clarify"], str(statuses)))
    # Failure search plumbing (no physics): plausibility gate and a valid stand-in proposal.
    ok_in, _ = plausible({"template": "bowl", "params": {"radius": 0.07}, "pos": [0.5, 0.0]}, 0)
    ok_out, why = plausible({"template": "bowl", "params": {"radius": 0.07}, "pos": [0.68, 0.32]}, 0)
    prop = client.complete("search_proposal", {"batch": 16, "budget_left": 0, "space": SPACE, "xy": XY,
                                               "stats": {}, "failures": [], "successes": []}).output
    results.append(check("failure search plumbing", ok_in and not ok_out and
                         sum(r["count"] for r in prop["regions"]) == 16, f"rejects corner: {why}"))
    # Orientation of a square top: a large block rotated 30 deg must be perceived within 8 deg (mod 90).
    import math
    sc = build_scene(probe(3, {"template": "block", "params": {"size": 0.062}, "pos": [0.5, 0.0],
                               "yaw": math.radians(30)}), seed=3)
    yaw = Perception(sc, "sim").perceive(["block"])["block"]["yaw"]
    yerr = abs((math.degrees(yaw) - 30 + 45) % 90 - 45)
    results.append(check("block orientation", yerr < 8, f"error={yerr:.1f} deg"))

    # Retrieval on the rule backend picks a valid library skill for a mug.
    ret = client.complete("retrieve", {"object": {"width": 0.15, "height": 0.1, "hollow": True, "rim_radius": 0.045,
                                                  "shape_group": "hollow-wide-tall"},
                                       "candidates": [{"card": "rim_pinch", "description": "rim grasp",
                                                       "requires": {"hollow": True, "rim": True},
                                                       "evidence": {"hollow-wide-tall": "4/4"}, "found_on": "bowl"},
                                                      {"card": "side_wrap", "description": "side grasp", "requires": {},
                                                       "evidence": {}, "found_on": "seed"}]}).output
    results.append(check("retrieval (rule backend)", ret["choice"] == "rim_pinch", ret["explanation"]))

    # Plan programs: the static check refuses imports; the stand-in program stacks three blocks.
    bad = check_program("import os\nos.remove('x')")
    stack = make_task("stack_by_size", seed=0)
    prog = client.complete("plan_program", {"prompt": stack["prompt"], "task": stack}).output["program"]
    rec_p, _ = run_episode(stack, seed=0, program=prog)
    results.append(check("plan program (rule backend)", bool(bad) and rec_p["success"],
                         f"import refused: {bool(bad)}, stack: {rec_p['success']} {rec_p.get('program_error')}"))

    # Composite cards: signature by goal structure; a program with ids applies only where those ids exist.
    sig = signature(stack)
    card = {"signature": sig, "program": 'pick(find_one(id="block_small"))\nplace(find_one(id="block_xyz"))'}
    results.append(check("composite signature/applicability",
                         sig == ["on:block->block"] * 2 and applicable({**card, "program": prog}, stack)
                         and not applicable(card, stack), str(sig)))

    # Model-written primitive (rule stand-in): static check, then one plate slid onto its zone by a push card.
    prop = client.complete("propose_primitive", {"capability": "push", "why": "smoke", "obj": {}, "target": {}}).output
    pcard = {"name": prop["name"], "kind": "push", "version": 1, "impl": "generated", "code": prop["code"],
             "params": prop["params"]}
    rec_push, _ = run_episode(make_task("slide_plate", seed=0), seed=0, extra_cards={pcard["name"]: pcard},
                              program="push(find_one(id='plate'), find_one(id='zone'))")
    results.append(check("generated push primitive", not check_primitive(prop["code"]) and rec_push["success"],
                         f"{rec_push['subtasks'][0]['label']} {rec_push.get('program_error')}"))

    # Articulated object: the drawer's handle is perceived as a part, the stand-in opening motion (part-level
    # grasp_part + move_linear until the hand stalls) pulls a short drawer to its end stop, and the measured joint
    # goal (drawer slid out >= 8 cm) still holds at the end of the episode.
    dtask = make_task("open_drawer", seed=0)
    dtask["objects"][0]["params"]["travel"] = 0.09   # a short drawer: the pull must stop at the end stop
    dscene = build_scene(dtask, seed=0)
    dper = Perception(dscene, "sim")
    dparts = (dper.perceive()["drawer"] or {}).get("parts", {})
    dper.close()
    oprop = client.complete("propose_primitive", {"capability": "open_drawer", "why": "smoke",
                                                  "obj": {"parts": dparts}, "target": None}).output
    ocard = {"name": oprop["name"], "kind": "open_drawer", "version": 1, "impl": "generated", "code": oprop["code"],
             "params": oprop["params"]}
    rec_open, _ = run_episode(dtask, seed=0, extra_cards={ocard["name"]: ocard},
                              program="use('open_drawer', find_one(id='drawer'))")
    results.append(check("drawer: handle part + opening skill + joint goal",
                         "handle" in dparts and rec_open["success"],
                         f"{rec_open['subtasks'][0].get('label')} {rec_open.get('program_error')}"))

    # Feasibility before running: a plate fits no grasp (push covers it), a drawer needs an opening skill the base
    # library lacks; the oracle judges a drawer scene with its hand-written reference motions and true parts.
    from kintsugi.feasibility import assess
    psc = build_scene(make_task("slide_plate", seed=0), seed=0)
    pper = Perception(psc, "sim")
    pv = assess(make_task("slide_plate", seed=0), pper.perceive(), {**seed_library(), pcard["name"]: pcard})
    pper.close()
    dv = assess(dtask, Perception(dscene, "sim").perceive(), seed_library())
    od = run_oracle(dtask, seed=0)
    results.append(check("feasibility + oracle for motions", pv[0]["verdict"] == "motion" and dv[0]["verdict"] == "missing"
                         and od["success"], f"plate {pv[0]['verdict']}, drawer {dv[0]['verdict']}, oracle {od['success']}"))

    # Objects from files (objects/*.xml|*.urdf): a URDF item is perceived like any object; an MJCF cabinet's drawer is a
    # moving part whose opening skill is named after the part (open_drawer), so library skills carry over.
    from kintsugi.sim.templates import ASSET_TEMPLATES, capability_for
    imp_ok, imp_detail = True, "no files in objects/"
    if {"soap_bottle", "nightstand"} <= ASSET_TEMPLATES:
        st = {"name": "imp", "prompt": "", "constraints": [], "goal": [{"pred": "joint", "obj": "ns", "op": ">=", "value": 0.08}],
              "objects": [{"id": "soap", "template": "soap_bottle", "role": "item", "params": {}},
                          {"id": "ns", "template": "nightstand", "role": "item", "params": {}}]}
        sc = build_scene(st, seed=0)
        ps, pg = Perception(sc, "sim").perceive(), Perception(sc, "ground_truth").perceive()
        err = float(np.linalg.norm(np.subtract(ps["soap"]["center"], pg["soap"]["center"])))
        imp_ok = err < 0.015 and "handle" in (ps["ns"] or {}).get("parts", {}) and capability_for("nightstand", "open") == "open_drawer"
        imp_detail = f"soap centre err {1000 * err:.1f} mm, nightstand parts {list((ps['ns'] or {}).get('parts', {}))}"
    results.append(check("imported objects (MJCF + URDF)", imp_ok, imp_detail))
    # Scanned objects (optional download, scripts/fetch_assets.py --scanned): a textured mesh is perceived close to
    # its true shape and the library puts it on a tray.
    if {"pill_bottle", "mug_blue"} <= ASSET_TEMPLATES:
        sst = {"name": "scan", "prompt": "", "constraints": [],
               "objects": [{"id": "pills", "template": "pill_bottle", "role": "item", "params": {}},
                           {"id": "mug", "template": "mug_blue", "role": "item", "params": {}},
                           {"id": "tray", "template": "tray", "role": "target", "params": {}}],
               "goal": [{"pred": "on", "obj": "pills", "target": "tray"}]}
        sc = build_scene(sst, seed=0)
        ps, pg = Perception(sc, "sim").perceive(), Perception(sc, "ground_truth").perceive()
        errs = [float(np.linalg.norm(np.subtract(ps[o]["center"], pg[o]["center"]))) for o in ("pills", "mug")]
        sr, _ = run_episode(sst, seed=0)
        results.append(check("scanned objects (meshes)", max(errs) < 0.01 and sr["success"] and ps["mug"]["hollow"],
                             f"centre err {[round(1000 * e, 1) for e in errs]} mm, mug hollow {ps['mug']['hollow']}, "
                             f"pills on tray {sr['success']}"))
    # Relational goal: the fixed plan puts a block to the left of a bowl (checked from positions).
    rel = {"name": "rel", "prompt": "", "constraints": [],
           "objects": [{"id": "block", "template": "block", "role": "item", "params": {"color": "red"}},
                       {"id": "bowl", "template": "bowl", "role": "target", "params": {}}],
           "goal": [{"pred": "left_of", "obj": "block", "target": "bowl"}]}
    rr, _ = run_episode(rel, seed=0)
    results.append(check("relational goal (left_of)", rr["success"], str([s_.get("label") for s_ in rr["subtasks"]])))
    # A bottle lying on its side: perceived with its length along the table (truth agrees), feasibility asks for
    # stand_up, and the stand-in turn (hand turned about the fingertip line) stands it back up.
    st_task = make_task("stand_up", seed=0)
    sc = build_scene(st_task, seed=0)
    dl, gl = Perception(sc, "sim").perceive(["bottle"])["bottle"], Perception(sc, "ground_truth").perceive(["bottle"])["bottle"]
    true_len = st_task["objects"][0]["params"]["height"]
    sprop = client.complete("propose_primitive", {"capability": "stand_up", "why": "smoke", "obj": {}, "target": {}}).output
    scard = {"name": sprop["name"], "kind": "stand_up", "version": 1, "impl": "generated", "code": sprop["code"],
             "params": sprop["params"]}
    rs, _ = run_episode(st_task, seed=0, extra_cards={scard["name"]: scard}, program="use('stand_up', find_one(id='bottle'))")
    rmiss, _ = run_episode(st_task, seed=0, feasibility=True, library=seed_library(), performance={}, patches={})
    results.append(check("lying bottle: perception + stand_up motion",
                         abs(dl["extent_major"] - true_len) < 0.01 and abs(gl["extent_major"] - true_len) < 0.01
                         and rs["success"] and rmiss.get("missing_capability") == "stand_up",
                         f"length {100 * dl['extent_major']:.1f} / truth {100 * gl['extent_major']:.1f} / "
                         f"{100 * true_len:.1f} cm, stood up {rs['success']}, missing {rmiss.get('missing_capability')}"))

    # Investigator on the rule backend: a tall cylinder fails with top_grasp; the shallow-grasp cause is confirmed.
    spec = probe(7, {"template": "cylinder", "params": {"radius": 0.025, "height": 0.14}, "pos": [0.5, 0.1], "yaw": 0.0})
    inv = Investigator(workers=1, library=seed_library(), performance={}, client=client, run_id="smoke/investigate")
    rep = inv.run([{"task_spec": spec, "seed": 7}])
    verdicts = [h.get("verdict") for c in rep["cases"] for h in c.get("hypotheses", [])]
    results.append(check("investigator (rule backend)", "confirmed" in verdicts, str(verdicts)))
    print(f"\n{sum(results)}/{len(results)} passed in {time.time() - t0:.0f}s. Images in {OUT}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
