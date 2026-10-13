# Reaction Mechanism Discovery Challenge -- repository contract.
# Judges run these four, in order, on a clean machine. Keep them working.

# Use the standard Windows shell without requiring a separate Unix shell.
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# install dependencies and the local package (editable)
install:
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

# write submission/predictions.csv for the held-out conditions
predict:
    python submission_workflow.py predict

# write submission/campaign.yaml (the experiments you would run next)
suggest:
    python submission_workflow.py suggest

# regenerate every figure used in the presentation
replicate:
    python submission_workflow.py replicate
