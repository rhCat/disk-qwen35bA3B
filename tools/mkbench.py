#!/usr/bin/env python3
# mkbench.py -- mock-SOP QA benchmark. Each scenario: a realistic
# document with ONE unique unguessable fact buried in it, plus a
# question. Graded by exact-fact extraction from the generated text.
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit")

# ---------------- scenario documents ----------------
scenarios = {}

scenarios["sop-cleanroom"] = {
    "doc": """STANDARD OPERATING PROCEDURE 0142 — CLEANROOM ENTRY
Purpose: Define the controlled entry procedure for ISO Class 5 cleanrooms.
Scope: All personnel entering the semiconductor fabrication cleanroom at Building 3.

1. Preparation. All personnel must don the full cleanroom suit: hood, goggles,
   face mask, two pairs of gloves, and booties. Suits are stored in the gowning
   room lockers and must be inspected for tears before use.
2. Air shower. Enter the air shower chamber and stand with arms raised. The
   chamber runs a 12-second cycle that removes loose particles from the suit
   surface. Do not open the inner door until the cycle completes and the green
   light illuminates.
3. Entry log. Sign the electronic entry log at the terminal beside the inner
   door. The log records badge number, time in, and time out. The system
   rejects badges that have not completed the gowning checklist.
4. Glove check. After entry, wipe gloves with isopropyl alcohol at the
   workstation near the inspection bench. Replace gloves immediately if any
   contamination is observed.
5. Exit. Exit through the same air shower. Used suits are placed in the
   designated laundry bins, never in general waste.

Emergency contact for cleanroom issues: the facility manager's office is
Room 318 in Building 3, and the after-hours line is extension 7714.

The cleanroom air handling unit is designated AHU-CL3 and its filter bank
must be replaced when differential pressure exceeds 0.9 inches of water.""",
    "question": "According to SOP 0142, how many seconds does the air shower cycle run before the inner door opens?",
    "answer": "12",
}

scenarios["sop-reactor"] = {
    "doc": """STANDARD OPERATING PROCEDURE 0221 — REACTOR COOLANT LOOP MAINTENANCE
Purpose: Maintenance of the primary coolant loop on reactor units R-1 through R-4.
Applicability: Maintenance technicians certified in loop operations, level 2 or above.

1. Isolation. Before any work, isolate the coolant loop by closing the primary
   isolation valve, designated VLV-77B, located on the south wall of the pump
   bay. Lock the valve handle with the standard red lockout tag.
2. Pressure verification. Verify loop pressure is below 5 psi using the digital
   gauge on the control panel. Never begin disassembly with pressure above this
   limit.
3. Drain. Open the drain port at the lowest point of the loop and collect the
   coolant in the approved drums. Dispose of drained coolant per environmental
   procedure EP-04.
4. Filter replacement. Remove the filter housing cover using the TWR-9 torque
   wrench set to 18 foot-pounds. Install the new filter element with the arrow
   pointing in the direction of flow.
5. Reassembly. Reinstall the housing cover and torque to 18 foot-pounds. Open
   the isolation valve and verify flow with the rotameter.
6. Documentation. Record the work in the maintenance log, including the filter
   lot number and the technician's badge number.

The spare filter elements are stored in the parts cage, shelf F, and are
catalogued under part number FLT-4482.""",
    "question": "What tool is specified for removing the filter housing cover in SOP 0221?",
    "answer": "TWR-9",
}

scenarios["ctx-warehouse"] = {
    "doc": """INTERNAL MEMO — DISTRIBUTION CENTER UPDATE
To: All distribution staff
From: Operations Manager
Date: June 14, 2026

This memo summarizes the changes taking effect at the Meridian Distribution
Center over the next quarter.

First, the receiving dock will be resurfaced during the week of July 6. All
inbound deliveries should use the alternate dock on the north side during
that period. The alternate dock has two bays and a smaller capacity, so
carriers should stagger arrival times.

Second, we are rolling out a new inventory system, WMS 5.2, starting July 20.
Training sessions will be held in the training room every Tuesday and
Thursday at 10:00 AM through August. All staff must complete the training
before they are granted system access.

Third, the cold storage area will be expanded to accommodate the growing
perishable goods volume. Construction begins August 3 and is expected to
take six weeks. During construction, perishable items will be staged in the
temporary refrigerated trailers behind the building.

Fourth, the security team has implemented a new visitor badge policy. All
visitors must exchange a government-issued ID for a visitor badge at the
front desk. The visitor badge allows access to the lobby, conference rooms,
and the cafeteria only. Access to the warehouse floor requires an escort.

Fifth, the backup generator, designated GEN-3300, will undergo its annual
maintenance during the week of September 14. The facility will run on
utility power during that period, and non-critical systems may see
interruptions.

Any questions about these changes should be directed to the operations
office at extension 8823.""",
    "question": "What is the designation of the backup generator mentioned in the memo?",
    "answer": "GEN-3300",
}

scenarios["ctx-negative"] = {
    "doc": """EMPLOYEE HANDBOOK EXCERPT — VELOCITY LOGISTICS
Section 5: Workplace Policies

5.1 Attendance. Employees are expected to arrive by 9:00 AM and record their
attendance using the time clock at the main entrance. Late arrivals must be
reported to the shift supervisor.

5.2 Breaks. Employees receive a 15-minute morning break and a 30-minute lunch
break. Break times are staggered by department to avoid crowding in the break
room.

5.3 Communication. Official announcements are posted on the bulletin board
outside the cafeteria and distributed by email. Employees should check both
channels daily.

5.4 Personal items. Personal electronic devices should be kept in lockers
during shifts. The company is not responsible for items left in common areas.

5.5 Safety. Safety shoes are required in all warehouse areas. Safety glasses
are required in the machining area. Hard hats are required on the loading
dock when overhead cranes are in operation.

5.6 Incident reporting. All workplace incidents must be reported to the
safety office within 24 hours. Incident forms are available at the safety
office and online.

5.7 Overtime. Overtime must be approved in advance by the department manager.
Approved overtime is paid at one and a half times the regular rate.""",
    "question": "According to the handbook, what is the exact fire evacuation assembly point?",
    "answer": "NOT IN DOCUMENT",
}

# ---------------- build prompts + ids ----------------
# Pad docs to varied context lengths (short/mid/long) with extra
# procedure sections so the bench spans the length curve.
PAD = {
    "sop-cleanroom": 0,      # ~400 tokens (short)
    "sop-reactor": 6,        # ~1.4K (mid)
    "ctx-warehouse": 11,     # ~2.2K (long)
    "ctx-negative": 0,       # ~330 (control)
}
PAD_TEXT = (
    "Standard maintenance checks are performed on a scheduled basis. "
    "Technicians follow the checklist printed in the appendix of each "
    "procedure. The checklist covers visual inspection, fastener "
    "verification, seal condition, and documentation of any anomalies. "
    "Any item failing inspection is flagged in the maintenance system "
    "and corrected before the equipment is returned to service. "
    "Records are retained for seven years per company policy. "
    "The facility operates on a continuous schedule with three shifts. "
    "Shift handover includes a review of open work orders and any "
    "equipment in a degraded state. Shift supervisors are responsible "
    "for ensuring the handover log is complete and signed. "
    "Periodic training refreshers are required every six months for all "
    "operating personnel. Training records are tracked in the learning "
    "management system and reviewed during annual audits. "
    "Unannounced audits occur at least once per quarter to verify "
    "compliance with documented procedures. Audit findings are tracked "
    "to closure by the quality team. "
    "All equipment has a unique asset tag affixed to the housing. The "
    "tag includes the asset number and the calibration due date. "
    "Calibration is performed by the metrology lab and certified "
    "against traceable standards. Out-of-calibration equipment is "
    "quarantined until recertification is complete."
)

meta = {}
for name, s in scenarios.items():
    doc = s["doc"]
    for _ in range(PAD.get(name, 0)):
        doc = doc + "\n\n" + PAD_TEXT
    user = (f"Read the following document and answer the question. "
            f"Answer with just the requested fact, no explanation.\n\n"
            f"DOCUMENT:\n{doc}\n\n"
            f"QUESTION: {s['question']}")
    prompt = (f"<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    ids = tok(prompt)["input_ids"]
    with open(f"/tmp/q35-bench-{name}.txt", "w") as f:
        f.write(",".join(str(i) for i in ids))
    meta[name] = {
        "question": s["question"],
        "answer": s["answer"],
        "prompt_tokens": len(ids),
    }
    print(f"{name}: {len(ids)} prompt tokens")

with open("/tmp/q35-bench-meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("wrote /tmp/q35-bench-meta.json")
