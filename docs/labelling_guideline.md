# 2026 Test Set – Labelling Guideline (DRAFT, Week 1)

Owner: Narinder Kaur · To be finalised by the team in Week 2 (lead: Arti Makwana)

## Purpose
We will collect about **200 recent job postings (2025–2026)**, about 50 per member, to test whether models trained on
EMSCAD (2012–2014) still detect current scams. This set is **for testing only, never for training**.

## Target mix
- About **100 fraudulent** and **100 legitimate** postings.
- Legitimate postings from several industries and job levels, not only tech.

## Label definitions
| Label | Use when |
|---|---|
| **1 = Fraudulent** | The posting is confirmed as a scam by a reliable source, **or** it shows at least **two strong red flags** below |
| **0 = Legitimate** | Posted by a verifiable employer (real company website/careers page) on a mainstream job board, with no strong red flags |
| **Exclude** | Not enough information to decide – do not guess |

### Strong red flags
- Asks the applicant to pay (training, equipment, background check, "starter kit")
- Asks for SIN, bank details or ID before any interview
- Interview only by WhatsApp, Telegram or text chat
- Payment by cryptocurrency, gift cards or a cheque to deposit and send back
- Pay far above normal for the role with no experience needed ("earn $500/day from home")
- Company cannot be found online, or uses a personal e-mail address (Gmail, Yahoo, Outlook)

## Allowed sources
- **Fraudulent:** Reddit r/Scams posts showing the posting text, Canadian Anti-Fraud Centre and Better Business Bureau Scam Tracker examples
- **Legitimate:** Canada Job Bank, LinkedIn, Indeed, company careers pages

## What to record (shared Google Sheet)
| Column | Description |
|---|---|
| id | JS26-001, JS26-002, … |
| collector | Member name |
| date_collected | YYYY-MM-DD |
| source | Website / subreddit |
| url | Link to the posting (if available) |
| title, company_profile, description, requirements, benefits | Copy the posting text into the same fields as EMSCAD |
| salary_range, location, employment_type | If shown |
| label | 1, 0 or exclude |
| evidence | Which red flags / confirmation source |
| second_reviewer | Another member who checked the label |

## Quality control
- Every label is checked by **a second member**; disagreements are discussed and decided by the week's lead.
- Remove personal information (names, phone numbers, e-mails of individuals) before saving.
