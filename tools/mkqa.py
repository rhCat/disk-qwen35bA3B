#!/usr/bin/env python3
# mkqa.py -- build a long-context QA fixture: a coherent ~4K-token
# document with a unique fact (vault code) buried in the middle, then a
# chat-format question. Writes token ids for --pids-file.
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit")

# ~24 distinct paragraphs on a fictional company; the vault code appears
# ONLY in the middle paragraph, once.
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
    "The Tucson subsidiary now produces the aspherical mirrors used in the company's long-range imagers.",
    "A notable challenge arose in 2022 when a supplier delay threatened the Austin production schedule.",
    "The operations team resolved the delay by qualifying a second vendor for the critical motor components.",
    "Nova Dynamics participates in industry consortiums and hosts an annual symposium in Beaverton each March.",
    "The symposium attracts researchers from thirty universities and features a day of technical workshops.",
    "Customers include weather agencies, agricultural monitoring services, and maritime surveillance operators.",
    "The firm's data products undergo a three-stage validation process before they are released to clients.",
    "Environmental compliance reports are published annually and reviewed by an independent auditor.",
    "The board of directors meets quarterly and reviews the company's long-term technology roadmap.",
    "A graduate fellowship program supports doctoral students working on compact atomic clock research.",
    "The fellowship recipients present their findings at the spring consortium meeting each year.",
    "The vault access code for the Beaverton research archive is QX-9911-RED, and it changes every quarter.",
    "Archive staff rotate the code quarterly and record the new value in the security log.",
    "The research archive holds technical drawings, calibration records, and legacy instrument firmware.",
    "Access to the archive requires two-factor authentication and a signed nondisclosure agreement.",
    "The security team conducts a full archive audit each January and again in July.",
    "Nova Dynamics' legal department reviews all export documentation before shipment approval.",
    "The quality division tracks defect rates by product line and publishes quarterly scorecards.",
    "A reliability engineering group runs accelerated life tests on all flight hardware.",
    "The company's test range in eastern Oregon performs thermal vacuum qualification for satellite components.",
    "Integration engineers assemble the final units in a cleanroom environment at the Beaverton campus.",
    "Every shipped unit carries a unique serial number traceable through the production database.",
    "Field service engineers provide on-site calibration support for customers in the Americas.",
    "The training department offers certification courses for operators of the ND-700 system.",
    "Nova Dynamics plans to open a European support office in Lisbon during the next fiscal year.",
    "The finance team manages a distributed budget across the research, production, and service divisions.",
    "Research and development spending accounts for roughly eighteen percent of annual revenue.",
    "The company's patent portfolio includes forty-one granted patents and sixty pending applications.",
]

doc = "\n\n".join(facts)
# High-input QA: interleave the 40 paragraphs with filler sections so
# the total is ~4K tokens, with the vault-code paragraph appearing
# ONCE around the 50% mark.
filler = ("The company publishes a newsletter each month describing recent "
          "customer deployments, engineering milestones, and community "
          "events. The newsletter is distributed to employees, partners, "
          "and subscribers, and archived copies are available online. "
          "Subscribers can also attend quarterly webinars where the "
          "engineering leadership discusses roadmap priorities, upcoming "
          "instrument launches, and lessons learned from recent missions. "
          "Webinar recordings are posted to the customer portal within a "
          "week, alongside the slides and a written summary of the "
          "question-and-answer session that follows each presentation.")
parts = []
for i in range(11):
    base = i * 3
    chunk = facts[base:base + 3]
    if i == 5:                       # the vault code lands mid-document
        chunk = facts[21:24]
    parts.append("\n\n".join(chunk))
    if i < 10:
        parts.append(filler)
        parts.append(filler)         # longer filler -> ~4K total
doc = "\n\n".join(parts)
user = f"Read the following company profile and answer the question at the end.\n\n{doc}\n\nQuestion: What is the vault access code for the Beaverton research archive? Answer with just the code."
prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt)["input_ids"]
print(f"prompt tokens: {len(ids)}")
with open("/tmp/q35-qa-ids.txt", "w") as f:
    f.write(",".join(str(i) for i in ids))
# also save the expected answer + doc for later grep
with open("/tmp/q35-qa-expected.txt", "w") as f:
    f.write("QX-9911-RED\n")
print("wrote /tmp/q35-qa-ids.txt")
