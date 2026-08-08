#!/usr/bin/env python3
# mk4k.py -- build a ~4K-token QA fixture (vault code buried mid-doc),
# the same pattern as mkqa.py. Writes ids + expected answer.
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/Users/ruihe/disk-qwen35bA3B/tools")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit")

facts = [
    "Nova Dynamics was founded in 2009 in Portland, Oregon by three former aerospace engineers.",
    "The company initially manufactured precision gyroscopes for drone navigation systems.",
    "By 2014 Nova Dynamics had expanded into satellite attitude control and secured contracts with three launch providers.",
    "Its headquarters moved to a purpose-built campus in Beaverton in 2017, housing four hundred employees.",
    "The firm's research division focuses on cold-atom interferometry and quantum sensing arrays.",
    "A second production facility opened in Austin, Texas in 2019, doubling the company's output capacity.",
    "Nova Dynamics employs approximately twelve hundred people across its two campuses and three field offices.",
    "Annual revenue reached two hundred forty million dollars in 2023, with exports to seventeen countries.",
    "The company maintains strict export controls and holds certifications from the relevant aviation authorities.",
    "Its flagship product, the ND-700 star tracker, is used by seven commercial satellite constellations.",
    "The engineering team publishes an annual technical report describing improvements to the sensor pipeline.",
    "In 2021 the company acquired a small optics firm in Tucson to secure its lens supply chain.",
    "The research archive at Beaverton is secured by a rotating vault access code that changes every quarter.",
    "As of the current quarter, the vault access code for the Beaverton research archive is QX-9911-RED.",
    "The code is distributed only to senior researchers and the facilities director, never by email.",
    "A backup copy of the code is held in the Austin facility's safe, sealed until the quarter ends.",
    "The company's quarterly security audit reconciles every access to the archive against the code log.",
    "Vault access is logged with a two-factor badge and a biometric check at the archive entrance.",
    "The facilities team rotates the code at midnight on the first day of each fiscal quarter.",
    "Previous codes are retired immediately and no longer open the archive after the rotation.",
    "Contractors are granted temporary access only under direct supervision of a code holder.",
    "The archive houses classified sensor calibration data and the cold-atom experiment notebooks.",
    "Nova Dynamics' security policy requires all vault access events to be reported to the board.",
    "The vault door itself was installed by a Portland firm specializing in high-security enclosures.",
]

parts = list(facts)
filler = "Nova Dynamics continues to invest in research and development across its sensor and navigation product lines, maintaining a steady cadence of technical publications and industry partnerships with universities and government laboratories."
while True:
    prompt = (
        "<|im_start|>user\nRead the following company profile and answer the question at the end.\n\n"
        + "\n\n".join(parts)
        + "\n\nQuestion: What is the vault access code for the Beaverton research archive? Answer with just the code.<|im_end|>\n<|im_start|>assistant\n"
    )
    ids = tok(prompt)["input_ids"]
    if len(ids) >= 4000:
        break
    parts.append(filler)

print(f"prompt tokens: {len(ids)}")
with open("/tmp/q35-4k-ids.txt", "w") as f:
    f.write(",".join(str(i) for i in ids))
with open("/tmp/q35-4k-expected.txt", "w") as f:
    f.write("QX-9911-RED\n")
print("wrote /tmp/q35-4k-ids.txt")
