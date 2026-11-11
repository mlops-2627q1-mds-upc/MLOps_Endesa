# EDN-022: Fit the Virtech deployment to the account quota

Date: 2026-11-11 (recorded).

Milestone: M4

Activity / Topic: Deployment target (#75)

Decision Participants: Dídac Cayuela

Status: Recorded; peer review pending.

## Decision

I chose Virtech with private SSH access to a loopback API. I selected the 20 GiB
template 108 for VM 2245, with explicit image/model identities and read-only model
files.

## Rationale

The account allowed 24 GiB of system disk, so the 30 GiB Ubuntu 24.04 template did not
fit. Template 108 fitted without requesting more storage, at the cost of using Ubuntu
20.04.6. A separate benchmark container keeps measurement failures away from the main
service.

## AI Involvement

Information seeking; Alternative generation; Alternative assessment; Recommendation;
Solution generation; Other. I used Codex 6.1 Sol Max to compare templates against the
quota and implement and check the deployment I selected.

## Response to AI

Used as input for further analysis.

## Assessment of the AI Contribution

I used the quota comparison to choose template 108. The HTTP and lifecycle checks
provide evidence for that candidate configuration, including missing-model failure and
recovery.

## Other Evidence

[PR #94](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/94). The recorded
deployment checks date to 2 November; they do not establish later live availability.
