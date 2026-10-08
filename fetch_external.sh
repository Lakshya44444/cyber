#!/usr/bin/env bash
# Download the external Indian test sets into data/external (evaluation only).
set -e
mkdir -p data/external && cd data/external
tmp=$(mktemp -d)
git clone -q --depth 1 https://github.com/princebari/-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-Collection-of-Hindi-and-English-Messages.git "$tmp/iiitd"
cp "$tmp/iiitd/indian_spam.csv" iiitd_2011.csv
git clone -q --depth 1 https://github.com/junioralive/india-spam-sms-classification.git "$tmp/tel"
cp "$tmp/tel/dataset/spam_ham_india.csv" india_telecom_2024.csv; cp "$tmp/tel/LICENSE" LICENSE_india-telecom_MIT
git clone -q --depth 1 https://github.com/Rishabh-bgp/spam-scan.git "$tmp/ss"
cp "$tmp/ss/probes.py" spamscan_probes.py.txt; cp "$tmp/ss/examples.json" .; cp "$tmp/ss/LICENSE" LICENSE_spam-scan_MIT
rm -rf "$tmp"; echo "external data ready"
