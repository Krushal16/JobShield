# 2026 Test Set – Labelling Guideline (v1.0, Week 2)

Owner: Narinder Kaur · Approved by: Krushal Sardhara, Arti Makwana, Diksha Sharma, Narinder Kaur

## Purpose
We collect about **200 recent job postings (2025–2026)**, about 50 per member, to test whether models trained on
EMSCAD (2012–2014) still detect current scams. This set is **for testing only – never used for training or tuning**.

## Target mix and schedule
- About **100 fraudulent** and **100 legitimate** postings.
- Legitimate postings from several industries and job levels, not only tech.
- **Week 2 – pilot:** each member collects 5 postings (about 3 fake, 2 real) to test these rules.
- **Weeks 4–7 – main collection:** about 12 postings per member per week; finished by the end of Week 7.

## Label definitions
| Label | Use when |
|---|---|
| **1 = Fraudulent** | The posting is confirmed as a scam by a reliable source, **or** it shows at least **two strong red flags** |
| **0 = Legitimate** | Posted by a verifiable employer (real company website or careers page) on a mainstream job board, with no strong red flags |
| **exclude** | Not enough information to decide – do not guess |

### Strong red flags
1. Asks the applicant to pay (training, equipment, background check, "starter kit")
2. Asks for SIN, bank details or ID before any interview
3. Interview only by WhatsApp, Telegram or text chat
4. Payment by cryptocurrency, gift cards, or a cheque to deposit and send money back
5. Pay far above normal for the role with no experience needed ("earn $500/day from home")
6. Company cannot be found online, or the recruiter uses a personal e-mail address (Gmail, Yahoo, Outlook)

### Weak red flags (record them, but they never decide the label alone)
No company logo or profile · no salary · vague duties · urgent hiring language · many spelling mistakes

## Allowed sources
- **Fraudulent:** Reddit r/Scams posts that show the posting text, Canadian Anti-Fraud Centre and Better Business Bureau Scam Tracker examples
- **Legitimate:** Canada Job Bank, LinkedIn, Indeed, company careers pages

## Special cases
- **Screenshots:** if the posting appears only as an image, type the text into the sheet word for word. If the image is
  unreadable or only partly visible, label it **exclude**.
- **Messages, not postings:** a recruiter's private text or e-mail with no job advertisement is **excluded**
  (our model scores job postings, not chats).
- **Duplicates:** search the sheet for the job title before adding; do not add the same posting twice.
- **Non-English postings:** exclude.

## What to record (shared Google Sheet – columns match `docs/2026_test_set_template.csv`)
| Column | Description |
|---|---|
| id | JS26-001, JS26-002, … |
| collector | Member name |
| date_collected | YYYY-MM-DD |
| source / url | Website or subreddit, and link if available |
| title, company_profile, description, requirements, benefits | Copy the posting text into the same fields as EMSCAD (leave empty if not shown) |
| location, salary_range, employment_type | If shown |
| has_company_logo, telecommuting | 1 = yes, 0 = no |
| label | 1, 0 or exclude |
| red_flags | Numbers of the strong red flags present, e.g. "1;3" |
| evidence | Confirmation source or short reason |
| second_reviewer / agreed | Member who checked the label; yes or no |

## Quality control
- Every label is checked by **a second member**. If they disagree, that week's lead decides; if still unclear, **exclude**.
- Remove personal information (names, phone numbers and e-mails of individuals) before saving.
- After the pilot, the team reviews the 20 pilot postings together and updates this guideline if needed.
