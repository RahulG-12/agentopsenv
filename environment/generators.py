"""
AgentOpsEnv — Procedural generators for emails and tasks.
"""
from __future__ import annotations
import random
from typing import List, Tuple
from environment.models import Email, Task, EmailType, Priority, TaskStatus

_SPAM: List[Tuple[str, str, str]] = [
    ("You WON a $500 gift card!", "Claim your prize now by clicking this link. Limited time offer!", "promo@spamco.xyz"),
    ("Exclusive Investment Opportunity", "Dear Friend, I am a prince with $10M to share with you…", "prince@scammail.ng"),
    ("Account Compromised — Act Now!", "Verify your bank details immediately to restore access.", "security@notreal.ru"),
    ("Free iPhone 15 Pro", "You are our 1,000,000th visitor! Collect your phone today.", "winner@fakewins.io"),
    ("Lose 20kg in 7 Days!", "Our miracle supplement — buy 3 get 10 FREE. No exercise needed!", "health@scam.biz"),
    ("Invoice UNPAID #884421", "Wire $800 to settle your account immediately or face penalties.", "billing@fraudco.cc"),
    ("Pre-Approved Loan $50,000", "No credit check. Cash in 24 hours. Reply now.", "loans@instant.fake"),
]

_INFO: List[Tuple[str, str, str]] = [
    ("Q3 Company Newsletter", "Hi Team, Q3 highlights: revenue up 12%, new Bengaluru office, annual hackathon coming!", "hr@company.com"),
    ("IT Maintenance — Sunday 2AM", "Scheduled downtime Sunday 2–4 AM for server upgrades. Please save work.", "it@company.com"),
    ("All-Hands Meeting Invite", "All-hands this Friday at 3 PM. Agenda: roadmap, hiring, Q4 OKRs.", "ceo@company.com"),
    ("Team Lunch — Wednesday", "Team lunch Wednesday 1 PM at Spice Garden. RSVP to HR by Tuesday.", "hr@company.com"),
    ("Policy Update: Remote Work", "Remote work days capped at 3/week effective next month. See attached.", "admin@company.com"),
    ("Congrats to Priya on Promotion!", "Please join us in congratulating Priya — promoted to Senior Engineer!", "hr@company.com"),
    ("Employee Satisfaction Survey", "5-minute survey. Link expires in 2 weeks. Your feedback matters!", "survey@company.com"),
    ("Office Closed — Public Holiday", "Office closed Monday for Republic Day. Enjoy the long weekend.", "admin@company.com"),
]

# (subject, body, sender, task_title, effort, priority, rel_deadline_steps)
_ACTIONABLE: List[Tuple[str, str, str, str, int, Priority, int]] = [
    (
        "URGENT: Client Demo Prep Needed",
        "We have a critical client demo on Thursday. Please prepare a 10-slide deck covering Q2 metrics and our product roadmap. The client is Acme Corp — our most important account worth ₹2Cr.",
        "manager@company.com", "Prepare Acme Corp demo deck", 3, Priority.CRITICAL, 4
    ),
    (
        "CRITICAL: Login Failure on Production",
        "Users are getting 401 errors on the login page since the last deploy. This is blocking ~200 users. Needs immediate investigation and a hotfix deployed ASAP.",
        "devops@company.com", "Fix login 401 bug on prod", 2, Priority.CRITICAL, 2
    ),
    (
        "Budget Proposal Due Friday",
        "Please submit your department's FY2025 budget proposal by Friday EOD. Use the template on Google Drive. Finance needs this to finalize board presentation.",
        "finance@company.com", "Submit FY2025 budget proposal", 2, Priority.HIGH, 5
    ),
    (
        "Code Review Request: PR #142",
        "Hey, could you review my PR? It refactors the auth module — should be straightforward. Currently blocking the staging deployment and the team is waiting.",
        "dev@company.com", "Review PR #142 auth refactor", 1, Priority.HIGH, 3
    ),
    (
        "Onboard New Team Member — Monday",
        "Rahul joins Monday. Please set up his dev environment, GitHub access, Slack, Jira, and give him a codebase walkthrough. HR needs confirmation by EOD Friday.",
        "hr@company.com", "Onboard new engineer Rahul", 2, Priority.MEDIUM, 6
    ),
    (
        "Weekly Status Report Reminder",
        "Reminder: Please send your weekly status report to the PM by Thursday noon. Include blockers, progress percentage, and next steps. This goes into the board deck.",
        "pm@company.com", "Write weekly status report", 1, Priority.MEDIUM, 4
    ),
    (
        "Security Audit — Action Required",
        "Based on last week's audit, please rotate your API keys, update the secrets vault, and confirm to the security team by next Wednesday. Non-compliance risks a penalty.",
        "security@company.com", "Rotate API keys and update secrets vault", 1, Priority.HIGH, 7
    ),
    (
        "Design Review: New Analytics Dashboard",
        "The design team needs your feedback on the new analytics dashboard Figma mockups. Please review and add comments by EOW so we can move to dev sprint.",
        "design@company.com", "Review analytics dashboard mockups in Figma", 1, Priority.MEDIUM, 5
    ),
    (
        "Performance Review Self-Assessment Due",
        "It's that time — please complete your self-assessment in Workday by end of month. Your manager needs it 3 days before your 1:1 review scheduled for the 28th.",
        "hr@company.com", "Complete performance self-assessment", 2, Priority.LOW, 12
    ),
    (
        "API Documentation Out of Date",
        "The API docs are outdated after last sprint's changes. The partner integration team is blocked. Please update docs to reflect new endpoints before their call Friday.",
        "tech-lead@company.com", "Update API documentation for new endpoints", 2, Priority.MEDIUM, 8
    ),
    (
        "Database Migration Script Review",
        "The DB migration script for the PostgreSQL upgrade needs a second pair of eyes before we run it on prod Saturday. Can you review by Friday 5 PM?",
        "dba@company.com", "Review DB migration script for prod", 1, Priority.HIGH, 4
    ),
    (
        "Client Feedback Report — Q3",
        "Please compile the Q3 client feedback from Zendesk and prepare a summary report for the sales team. They need it for Monday's quarterly review.",
        "sales@company.com", "Compile Q3 client feedback report", 2, Priority.MEDIUM, 6
    ),
]

_AMBIGUOUS: List[Tuple[str, str, str]] = [
    ("Re: The thing from yesterday", "Hey, any update on that? Let me know. Thanks.", "colleague@company.com"),
    ("Following up", "Just checking in on this. We should sync this week if possible.", "stakeholder@partner.com"),
    ("Quick question about the project", "Hi, I had a few thoughts. Could we discuss? Might be an action item. Or not.", "director@company.com"),
    ("FYI", "Thought you should know. Might need something from you — TBD.", "random@company.com"),
    ("Re: Re: Re: Meeting", "Let's loop back on this. I think there might be a deliverable here somewhere.", "exec@company.com"),
]


def generate_email_set(difficulty: str = "medium", seed: int = 42, current_step: int = 0, max_steps: int = 30) -> List[Email]:
    rng = random.Random(seed)
    emails: List[Email] = []

    cfg = {
        "easy":   dict(spam=4, info=3, actionable=3, ambiguous=0),
        "medium": dict(spam=2, info=3, actionable=6, ambiguous=2),
        "hard":   dict(spam=2, info=2, actionable=10, ambiguous=3),
    }[difficulty]

    for tmpl in rng.sample(_SPAM, min(cfg["spam"], len(_SPAM))):
        subj, body, sender = tmpl
        emails.append(Email(subject=subj, body=body, sender=sender,
            priority=Priority.LOW, deadline=None, email_type=EmailType.SPAM,
            noise_level=0.05))

    for tmpl in rng.sample(_INFO, min(cfg["info"], len(_INFO))):
        subj, body, sender = tmpl
        emails.append(Email(subject=subj, body=body, sender=sender,
            priority=Priority.LOW, deadline=None, email_type=EmailType.INFO,
            noise_level=0.15))

    actionable_pool = rng.sample(_ACTIONABLE, min(cfg["actionable"], len(_ACTIONABLE)))
    for tmpl in actionable_pool:
        subj, body, sender, task_title, effort, priority, rel_dl = tmpl
        noise = 0.0
        noisy_body = body
        if difficulty == "hard":
            noise = rng.uniform(0.2, 0.55)
            fillers = [" [Fwd: original thread below]", " (see attachment — not attached)",
                       " — ping me if unclear!", " Thanks in advance!", " \n\nSent from my iPhone"]
            noisy_body += rng.choice(fillers)
            # Occasionally make deadline ambiguous
            if rng.random() < 0.3:
                rel_dl += rng.choice([-1, 1])
        # Scale deadline: min_dl is after realistic triage (read all + extract actionable)
        # Estimated triage = total_emails + num_actionable + buffer
        # This ensures deadlines are reachable by an efficient agent.
        # orig rel_dl range: 2 (urgent) → 12 (relaxed)
        # Map linearly to [60%, 92%] of max_steps, preserving relative urgency.
        t = (rel_dl - 2) / 10.0           # 0.0 = most urgent, 1.0 = least urgent
        min_pct = 0.60; max_pct = 0.92
        scaled_dl = int((min_pct + t * (max_pct - min_pct)) * max_steps)
        if difficulty == "hard" and rng.random() < 0.3:
            scaled_dl += rng.choice([-3, -2, 0, 2])  # deadline ambiguity
        scaled_dl = max(int(max_steps * 0.55), min(scaled_dl, max_steps - 2))
        emails.append(Email(
            subject=subj, body=noisy_body, sender=sender,
            priority=priority, deadline=current_step + scaled_dl,
            email_type=EmailType.ACTIONABLE,
            contains_task=True,
            task_keywords=[task_title.split()[0].lower(), priority.value],
            noise_level=noise,
        ))

    for tmpl in rng.sample(_AMBIGUOUS, min(cfg["ambiguous"], len(_AMBIGUOUS))):
        subj, body, sender = tmpl
        emails.append(Email(subject=subj, body=body, sender=sender,
            priority=Priority.MEDIUM, deadline=None,
            email_type=EmailType.AMBIGUOUS,
            contains_task=rng.random() < 0.25,
            noise_level=rng.uniform(0.55, 0.95)))

    rng.shuffle(emails)
    return emails


def generate_task_from_email(email: Email, current_step: int, seed: int = 42) -> Task:
    rng = random.Random(seed + hash(email.id) % 10000)
    for tmpl in _ACTIONABLE:
        subj, body, sender, task_title, effort, priority, rel_dl = tmpl
        if tmpl[0] == email.subject:
            return Task(
                title=task_title,
                description=body[:130] + "…",
                priority=priority,
                deadline=email.deadline or (current_step + rel_dl),
                effort=effort,
                status=TaskStatus.PENDING,
                source_email_id=email.id,
            )
    # Ambiguous fallback
    return Task(
        title=f"Follow-up: {email.subject[:45]}",
        description=email.body[:90] + "…",
        priority=Priority.LOW,
        deadline=current_step + 8,
        effort=1,
        status=TaskStatus.PENDING,
        source_email_id=email.id,
    )
