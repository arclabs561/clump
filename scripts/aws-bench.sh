#!/usr/bin/env bash
# aws-bench.sh -- orchestrate cross-architecture criterion benchmarks on AWS.
#
# Spins up one instance per architecture (Intel, AMD, Graviton3), runs
# benchmarks via aws-bench-remote.sh, collects criterion JSON, tears
# everything down, and produces a comparison table.
#
# Usage:
#   ./scripts/aws-bench.sh [--ref <git-ref>] [--dry-run]
#
# Prerequisites: aws cli v2, ssh, scp, python3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/arclabs561/clump"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TAG_VALUE="clump-bench-${TIMESTAMP}"
GIT_REF="HEAD"
DRY_RUN=false
SSH_USER="ec2-user"
RESULTS_DIR="${SCRIPT_DIR}/../target/bench-results/${TIMESTAMP}"

# -- argument parsing --------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)   GIT_REF="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# -- instance definitions ----------------------------------------------------
# Format: "label instance_type arch ami_name_pattern"
# Amazon Linux 2023 AMIs are region-specific; we resolve them by name.

INSTANCE_SPECS=(
    "intel_x86  c6i.xlarge x86_64  al2023-ami-2023.*-x86_64"
    "amd_x86    c6a.xlarge x86_64  al2023-ami-2023.*-x86_64"
    "graviton3  c7g.xlarge arm64   al2023-ami-2023.*-arm64"
)

# -- cleanup resources -------------------------------------------------------

KEY_NAME=""
KEY_FILE=""
SG_ID=""
declare -a INSTANCE_IDS=()

cleanup() {
    echo "--- cleanup: terminating instances and removing resources ---"
    if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
        aws ec2 terminate-instances --instance-ids "${INSTANCE_IDS[@]}" \
            --output text >/dev/null 2>&1 || true
        echo "Waiting for instances to terminate..."
        aws ec2 wait instance-terminated --instance-ids "${INSTANCE_IDS[@]}" \
            2>/dev/null || true
    fi
    if [[ -n "${SG_ID}" ]]; then
        aws ec2 delete-security-group --group-id "${SG_ID}" 2>/dev/null || true
    fi
    if [[ -n "${KEY_NAME}" ]]; then
        aws ec2 delete-key-pair --key-name "${KEY_NAME}" 2>/dev/null || true
    fi
    if [[ -n "${KEY_FILE}" && -f "${KEY_FILE}" ]]; then
        rm -f "${KEY_FILE}"
    fi
    echo "Cleanup complete."
}

trap cleanup EXIT

# -- helpers -----------------------------------------------------------------

log() { echo "[$(date +%H:%M:%S)] $*"; }

resolve_ami() {
    local arch="$1" pattern="$2"
    aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=${pattern}" \
                  "Name=architecture,Values=${arch}" \
                  "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text
}

wait_ssh() {
    local host="$1" key="$2"
    local attempts=0
    while ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
                -i "${key}" "${SSH_USER}@${host}" true 2>/dev/null; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -ge 60 ]]; then
            echo "SSH to ${host} timed out after 5 minutes" >&2
            return 1
        fi
        sleep 5
    done
}

# -- validate credentials ---------------------------------------------------

log "Validating AWS credentials..."
CALLER_IDENTITY="$(aws sts get-caller-identity --output json)"
ACCOUNT_ID="$(echo "${CALLER_IDENTITY}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Account'])")"
REGION="$(aws configure get region || echo "us-east-1")"
log "Account: ${ACCOUNT_ID}, Region: ${REGION}"

# -- resolve git ref to a full SHA -------------------------------------------

RESOLVED_REF="$(git -C "${SCRIPT_DIR}/.." rev-parse "${GIT_REF}")"
log "Git ref: ${GIT_REF} -> ${RESOLVED_REF}"

# -- dry run check -----------------------------------------------------------

if ${DRY_RUN}; then
    log "DRY RUN: would launch the following instances:"
    for spec in "${INSTANCE_SPECS[@]}"; do
        read -r label itype arch pattern <<< "${spec}"
        ami="$(resolve_ami "${arch}" "${pattern}")"
        log "  ${label}: ${itype} (${arch}) AMI=${ami}"
    done
    log "DRY RUN: exiting without launching."
    exit 0
fi

# -- create ephemeral key pair -----------------------------------------------

KEY_NAME="clump-bench-${TIMESTAMP}"
KEY_FILE="$(mktemp /tmp/clump-bench-key-XXXXXX.pem)"
aws ec2 create-key-pair --key-name "${KEY_NAME}" \
    --query 'KeyMaterial' --output text > "${KEY_FILE}"
chmod 600 "${KEY_FILE}"
log "Created key pair: ${KEY_NAME}"

# -- create security group (SSH from caller IP only) -------------------------

MY_IP="$(curl -s https://checkip.amazonaws.com)"
VPC_ID="$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(aws ec2 create-security-group \
    --group-name "${TAG_VALUE}" \
    --description "Ephemeral SG for clump benchmarking" \
    --vpc-id "${VPC_ID}" \
    --query 'GroupId' --output text)"
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 22 \
    --cidr "${MY_IP}/32" >/dev/null
log "Security group ${SG_ID}: SSH from ${MY_IP}/32"

# -- launch instances --------------------------------------------------------

mkdir -p "${RESULTS_DIR}"

declare -A LABEL_TO_IID
declare -A LABEL_TO_IP

for spec in "${INSTANCE_SPECS[@]}"; do
    read -r label itype arch pattern <<< "${spec}"
    ami="$(resolve_ami "${arch}" "${pattern}")"
    if [[ "${ami}" == "None" || -z "${ami}" ]]; then
        echo "Could not resolve AMI for ${arch} with pattern ${pattern}" >&2
        exit 1
    fi
    log "Launching ${label}: ${itype} (${arch}) AMI=${ami}"
    iid="$(aws ec2 run-instances \
        --image-id "${ami}" \
        --instance-type "${itype}" \
        --key-name "${KEY_NAME}" \
        --security-group-ids "${SG_ID}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${TAG_VALUE}-${label}}]" \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
        --query 'Instances[0].InstanceId' --output text)"
    INSTANCE_IDS+=("${iid}")
    LABEL_TO_IID["${label}"]="${iid}"
    log "  Instance ${iid} launched."
done

# -- wait for all instances to be ready --------------------------------------

log "Waiting for instance-status-ok on all instances..."
aws ec2 wait instance-status-ok --instance-ids "${INSTANCE_IDS[@]}"
log "All instances passed status checks."

# -- resolve public IPs ------------------------------------------------------

for spec in "${INSTANCE_SPECS[@]}"; do
    read -r label _ _ _ <<< "${spec}"
    iid="${LABEL_TO_IID[${label}]}"
    ip="$(aws ec2 describe-instances --instance-ids "${iid}" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
    LABEL_TO_IP["${label}"]="${ip}"
    log "${label}: ${ip}"
done

# -- run benchmarks on each instance -----------------------------------------

run_on_instance() {
    local label="$1" ip="$2"
    local ssh_opts="-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -i ${KEY_FILE}"
    local remote_script="/home/${SSH_USER}/aws-bench-remote.sh"
    local result_dir="${RESULTS_DIR}/${label}"
    mkdir -p "${result_dir}"

    log "${label}: waiting for SSH..."
    wait_ssh "${ip}" "${KEY_FILE}"

    log "${label}: uploading remote script..."
    scp ${ssh_opts} "${SCRIPT_DIR}/aws-bench-remote.sh" "${SSH_USER}@${ip}:${remote_script}"

    log "${label}: running benchmarks (ref=${RESOLVED_REF})..."
    ssh ${ssh_opts} "${SSH_USER}@${ip}" \
        "chmod +x ${remote_script} && ${remote_script} '${REPO_URL}' '${RESOLVED_REF}'"

    log "${label}: downloading results..."
    scp -r ${ssh_opts} \
        "${SSH_USER}@${ip}:/home/${SSH_USER}/clump/target/criterion" \
        "${result_dir}/criterion"

    log "${label}: benchmarks complete."
}

# Run all instances in parallel, collect exit codes.
declare -A BG_PIDS
for spec in "${INSTANCE_SPECS[@]}"; do
    read -r label _ _ _ <<< "${spec}"
    ip="${LABEL_TO_IP[${label}]}"
    run_on_instance "${label}" "${ip}" &
    BG_PIDS["${label}"]=$!
done

FAILURES=0
for label in "${!BG_PIDS[@]}"; do
    if ! wait "${BG_PIDS[${label}]}"; then
        echo "FAILED: ${label}" >&2
        FAILURES=$((FAILURES + 1))
    fi
done

if [[ ${FAILURES} -gt 0 ]]; then
    echo "${FAILURES} instance(s) failed. Partial results in ${RESULTS_DIR}." >&2
fi

# -- compare results ---------------------------------------------------------

log "Generating comparison table..."
python3 "${SCRIPT_DIR}/compare-benches.py" "${RESULTS_DIR}"

log "Results directory: ${RESULTS_DIR}"
log "Done."
