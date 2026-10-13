# CODEBOOK V2.2 — EMOTIONS IN YOUTUBE COMMENTS ON THE ATHLETE DROWNING CRISIS

Version: 2.2 · Status: revised after pilot pass 01 and the v2 recode (n = 200, single coder). Not yet validated by independent human coders.
Used by: human coders and the LLM coder (identical definitions for both)
Unit of analysis: one YouTube comment or reply, coded on its own text
Theoretical basis: The codebook is grounded in the moral outrage constructs of Coombs and Tachkova (2024), extended with a socio-cultural lens (class and identity talk, values, emotion-action links). Codes marked [ADDED] are study additions not taken from Coombs and Tachkova (2024). Their conceptual support is drawn from the crisis communication literature on attribution of crisis responsibility (Coombs, 2007), publics' emotions in crises (Jin et al., 2012), and social identity (Tajfel & Turner, 1979), and the citations indicate conceptual support, not adoption of an existing coding scheme. Codes marked [NEW] were added in version 2; [NEW in 2.1] in version 2.1.

## GENERAL RULES

1. Code what the comment expresses or clearly implies, not what the situation "should" make someone feel.
2. When in doubt, code 0 (or NONE / NA).
3. Code the commenter's own stance. Quoting or reporting others' views without endorsing them = 0.
4. Values, class, and identity are coded only when explicitly invoked, not because the case involves them.
5. Interpret Filipino context: slang, sarcasm, religious expressions, "po/opo," regional languages.
6. Multiple codes can apply unless the code says "choose ONE."
7. Video source is not coded here — it is attached from the dataset.
8. [NEW] Replies are coded on their own text. Code everything the reply itself expresses. If its stance or target cannot be determined without the comment it answers (e.g., "you're absolutely true," "oo, tama," "its okay bro"), code STANCE = NEUTRAL and set NEEDS_CONTEXT = 1. Do not guess the parent's content.
9. [NEW] Emojis carry meaning. Code emotions from unambiguous emojis: 😡 🤬 😠 = ANGER; 😭 😢 💔 😔 = SADNESS; 🙏 alone after a death = CONDOLENCE. Ambiguous emojis (😂, 😅, 🙄 used alone) are not coded as emotions unless context makes them clear. Emoji text may appear as :enraged_face: etc.
10. [NEW] Anonymize in evidence fields. Replace @handles and user IDs with [USER].

## SECTION A. LANGUAGE [ADDED]

A1. LANGUAGE (choose ONE)

- ENGLISH: Entirely or almost entirely English
- FILIPINO: Entirely or almost entirely Filipino/Tagalog
- TAGLISH: Meaningful mix of English and Filipino within the comment
- OTHER_PH_LANGUAGE: Cebuano/Bisaya, Ilocano, Hiligaynon, etc. (alone or mixed)
- EMOJI [NEW]: Emoji only, no words
- OTHER: Any other language

Note: LANGUAGE is a register variable. It is not a proxy for the commenter's class or education.

## SECTION B. EMOTIONS (0/1 EACH)

B1. ANGER

- Definition: Anger, frustration, or hostility directed at someone.
- Include: "galit na galit ako," "nakakainis," "kapal ng mukha," "bwisit," insults and profanity aimed at a target, 😡 🤬.
- Exclude: Swearing with no target; general exclamations of shock.

B2. CONTEMPT

- Definition: Scorn, mockery, or looking down on the target.
- Include: Sarcastic praise, ridicule, "blah blah blah," "boo hoo," "pathetic," mocking nicknames ("king kamote").

B3. DISGUST

- Definition: Revulsion toward a person or act.
- Include: "nakakadiri," "nakakasuka," "sickening," "disgusting."
- Note: rare in pilot (2%); retained because it is part of the DOC outrage definition.

B4. SADNESS [ADDED]

- Definition: Grief, heartbreak, sorrow.
- Include: "sobrang sakit," "nakakaiyak," "huhuhu," 😭 💔 when about the loss.

B5. CONDOLENCE [ADDED]

- Definition: Ritual expressions of sympathy or prayer with no further elaboration.
- Include: "RIP," "condolence po," "praying for the family," "fly high," 🙏 alone.

B6. COMPASSION_VICTIMS (renamed from COMPASSION)

- Definition: Engages with the victims' or family's suffering beyond ritual phrases: imagines their situation, dwells on their pain, or wishes to relieve it. (Coombs & Tachkova, P4–P5)
- Include: "put yourself in her shoes," "ramdam ko yung nawalan," "pushed to their limits… can't even swim."
- Note: A comment can be both CONDOLENCE and COMPASSION_VICTIMS.

B7. SYMPATHY_OFFENDER [NEW] [ADDED]

- Definition: Sympathy, pity, or prayers for the coach or other accused parties.
- Include: "nakakaawa din ang coach," "we also pray for you, coach," "so sorry for your loss too."
- Exclude: Sarcastic sympathy ("boo hoo, sorry for you") → CONTEMPT + SARCASM.

B8. FEAR [ADDED]

- Definition: Worry about one's own or others' safety, especially children or students.
- Include: "natakot tuloy ako para sa anak ko," "ayoko na payagan anak ko."
- Note: rare in pilot sample (coach-apology comments); retained to test on other video types.

B9. JOY [NEW in 2.2] [ADDED]

- Definition: Expressions of happiness, relief, or satisfaction.
- Include: "Mabuti nga," "I'm so happy," "finally," 🎉.
- Exclude: Sarcastic laughter (😂, 😅) directed at a target as an attack or mockery → code as CONTEMPT.

B10. SURPRISE [NEW in 2.2] [ADDED]

- Definition: Shock, disbelief, or astonishment regarding the incident, the revelations, or the official response.
- Include: "Hindi ako makapaniwala," "gulat ako," "shocking," 😮, 😱.
- Exclude: General rhetorical questions meant only to assign blame ("Bakit niyo ginawa yun?").

## SECTION C. MORAL APPRAISAL

C1. NORM_VIOLATION (0/1)

- Definition: Says or implies someone did something morally wrong — not merely unlucky.
- Include: "Mali 'yun," "irresponsable," "you failed those kids," "kasalanan mo."
- Exclude: "Tragic accident, walang may gusto."

C2. ATTRIBUTION (choose ONE) [ADDED] (scale)

- NONE
  Definition: No framing of cause or culpability
  Example: "RIP"
- ACCIDENT
  Definition: Nobody's fault; fate, God's will, tragedy
  Example: "Aksidente talaga, walang may gusto"
- ACCUSATION [NEW]
  Definition: Holds someone culpable for the deaths without specifying negligence or intent
  Example: "Kasalanan mo 'to," "killer coach," "you killed them," "rene is gone because of you"
- NEGLIGENCE
  Definition: Someone failed a duty of care: no lifeguard, unsafe location, poor supervision, carelessness
  Example: "Dapat may lifeguard," "reckless," "could have been prevented," "kapabayaan"
- INTENTIONAL
  Definition: Deliberate harm, cruelty, or foul play
  Example: "Pinarusahan hanggang mamatay," "hazing ang ikinamatay," "you intentionally inflict pain"
- VICTIM_FAULT
  Definition: The athletes themselves are blamed
  Example: "Are these guys adults? Did he force them?"

Decision rules:

- "Killer," "kasalanan mo," "you killed them" with no reason given = ACCUSATION.
- If the comment gives a reason, code the reason: carelessness/prevention → NEGLIGENCE; deliberate harm → INTENTIONAL.
- "Accident but could have been prevented" = NEGLIGENCE.
- Ignoring warnings is not coded here by itself — see C3. Code ATTRIBUTION from the comment's other language.

C3. IGNORED_WARNING [NEW] (0/1)

- Definition: Says the accused were warned (by resort staff, locals, police, signage) and proceeded anyway.
- Include: "Pinagsabihan na daw sya… pero nagmayabang," "the hotel staff warned you," "tinalo pa mga taga-doon na nagwarning."

C4. RECURRENCE [NEW] (0/1)

- Definition: Says a similar incident or harmful practice happened before and was not stopped.
- Include: "Nangyari na dati bakit inuulit pa," "you saved one of your players from drowning years ago," "it's been happening for years."

C5. COVER_UP (0/1) [ADDED]

- Definition: Accuses anyone of hiding facts, lying, or protecting reputation over truth.
- Include: "Pinagtatakpan nila," "he's still hiding something," "liar," "pinalabas nyo na nalunod."

C6. ALT_ACCOUNT [NEW] [ADDED] (0/1)

- Definition: Proposes a different account of how the deaths happened than the official one (hazing, initiation rites, foul play, "they didn't drown").
- Include: "Hazing ang ikinamatay," "hindi naniniwala na nalunod sila," "initiation rites."
- Note: often co-occurs with COVER_UP and INTENTIONAL; code each independently.

C7. INJUSTICE (0/1)

- Definition: Victims or family portrayed as treated unfairly or harmed undeservedly; justice not served.
- Include: "Hindi nila deserve 'yun," "walang hustisya," "there will be no justice."
- Exclude: "Justice for Rene" as a slogan alone → ACTIONS = ACCOUNTABILITY and VALUES = JUSTICE.

C8. SELF_INTEREST (0/1)

- Definition: Attributes the act or the response to self-interest: image, money, power, ego, careers, winning, or protecting the institution. (Adapts Coombs & Tachkova's "greed/exploitation.")
- Include: "Sad ka kasi mawawalan ka ng career," "coached by a legal team," "only aims to be great in basketball," "money is king."

C9. BLAME_TARGETS (list all)

- COACH
- COACHING_STAFF [NEW]
- UNIVERSITY
- PRESIDENT
- ORGANIZERS
- VENUE
- AUTHORITIES
- VICTIMS
- OTHER
- NONE

## SECTION D. STANCE TOWARD THE ORGANIZATION [ADDED]

D1. STANCE (choose ONE)

- CRITICAL
  Definition: Criticizes the coach, staff, university, president, or organizers
  Example: "Dapat managot ang coach"
- DEFENDING
  Definition: Defends them, excuses them, or rejects blaming them
  Example: "Prison? For an accident?"
- MIXED
  Definition: Both criticizes and defends
  Example: "Accountable, but don't treat him as a criminal"
- NEUTRAL
  Definition: No evaluative stance, or stance not determinable from the text
  Example: "RIP to both"

D2. NEEDS_CONTEXT [NEW] (0/1)
1 if the comment is a reply whose stance or meaning depends on the comment it answers.

## SECTION E. VALUES INVOKED (LIST ALL)

- FAMILY: Family bonds, parents, "anak," loss to the family
- CHILD_POTENTIAL: Youth, future, dreams, promise, "pag-asa," "dreamers," "puno ng pangarap"
- PARENT_SACRIFICE: Parents' hardship or sacrifice for the children
- DUTY_OF_CARE: Institutions must protect those entrusted to them ("in your care," "cargo ng eskwelahan")
- EDUCATION_MOBILITY: Schooling or athletics as a path out of poverty
- FAITH: God, prayer, religious meaning beyond ritual phrases
- JUSTICE: Fairness or justice as a value in itself
- SOLIDARITY: Community grief, "kapwa," "our finest players"
- SANCTITY_OF_LIFE [NEW]: Human life as sacred or not to be treated lightly
- OTHER: Name it in the evidence field
- NONE: No value invoked

## SECTION F. CLASS AND IDENTITY TALK [ADDED]

F1. CLASS_FRAME (0/1)

- Definition: Invokes social class, wealth, poverty, privilege, scholarship status, or unequal treatment of rich vs poor.
- Include: "Kahit madami pa kayo pera," "mahirap na probinsyano," "elite school sa Manila," "money is king," "classism, burgis."

F2. CLASS_DIRECTION (choose ONE; NA if CLASS_FRAME = 0)

- INEQUALITY_JUSTICE: The poor get less justice or protection; money buys impunity
- VICTIMS_POOR: Emphasizes the victims' humble background
- INSTITUTION_ELITE: Frames the university or officials as privileged or powerful
- OTHER: Other class framing
- NA: No class frame

F3. IDENTITY_FRAME [NEW] (list all)
Invokes a group identity of the accused, victims, or commenters.

- NATIONALITY
  Definition: Foreigner vs Filipino; deportation; "go home"
  Example: "Return home where you belong," "foreigners get forgiven," "fil people"
- RACE
  Definition: Race or skin color
  Example: "White boy"
- REGION
  Definition: Region or province vs capital
  Example: "Kami mga Mindanawon," "probinsyano," "school sa Manila"
- RELIGION
  Definition: Religious identity of the institution or actors
  Example: "In a Catholic school," "Jesuit priests"
- NONE: No identity invoked

F4. APPEARANCE_ATTACK [NEW] (0/1)

- Definition: Insults or mocks the accused's physical appearance.
- Include: "Kalbo," "opaw/upaw," "bald," "kalbohydrates."

## SECTION G. ACTION DEMANDS (LIST ALL)

- PUNITIVE: Legal or institutional punishment: jail, charges, firing, resignation, license revoked, deportation, bans, "managot," "kasuhan"
- RETALIATION [NEW]
  Definition: Wishes harm or suffering on the accused or their family outside legal channels
  "Dapat malunod din," "sana mangyari din sa anak mo," "he should swim there"
- RESTITUTION [NEW]
  Definition: Compensation or care for the families paid by the accused or institution
  "Pabayarin 10M each," "must pay the families' lifetime needs," "take care of the family"
- ACCOUNTABILITY: Investigation, judicial process, "#JusticeFor…," "face the families," "sagutin nila"
- INFORMATION [NEW]
  Definition: Demands the full facts: what happened, who decided
  "Tell me exactly what happened," "sabihin ang detalye"
- REFORM: Safety rules, policy change, preventing recurrence
- BOYCOTT: Avoid or withdraw from the university; sponsors withdraw; don't enroll
- SUPPORT: Donations or concrete help for the family from the public
- NONE: No action demanded

Rule: "praying for the family" alone = CONDOLENCE, not SUPPORT.

## SECTION H. RESPONSE EVALUATION (COOMBS & TACHKOVA P6–P10)

Code only if the comment refers to a specific response; otherwise NA.

H1. RESPONSE_TARGET (choose ONE)

- UNIVERSITY_ANNOUNCEMENT
- COACH_APOLOGY
- PRESIDENT_STATEMENT
- POLICE_REPORT
- OTHER
- NA

H2. RESPONSE_EVAL (choose ONE)

- ACKNOWLEDGES: Accepts it as admitting the wrong or as sincere
- REJECTS: Rejects it for any reason
- MIXED: Both
- NA: No response evaluated

H3. REJECT_REASONS [NEW] (list all; NA unless RESPONSE_EVAL = REJECTS or MIXED)

- INSINCERE
  Definition: Performance, acting, scripted, drama, crocodile tears, PR or legal coaching
  Example: "Scripted," "stop acting," "coached by a legal team"
- WORDS_NOT_ENOUGH
  Definition: Words cannot suffice; action or punishment is needed
  Example: "Sorry is not enough, go to jail"
- TOO_SOON
  Definition: Moving on / moving forward too early
  Example: "Kakamatay pa lang, 'move on' agad"
- TOO_LATE
  Definition: Apology came late
  Example: "Belated apologies," "why only speaking now?"
- NO_INFORMATION
  Definition: Gives no facts about what happened
  Example: "He gave us no information"
- SELF_CENTERED
  Definition: Centers the speaker's feelings over the victims
  Example: "Don't steal the grief," "you are not the victim here"
- SHIFTS_BLAME
  Definition: Blames others (resort, conditions)
  Example: "Don't blame it on the resort"
- DISMISSIVE [NEW in 2.1]
  Definition: Rejects the apology outright with no reason given
  Example: "Blah blah blah," "We will not accept your apology," "I don't care," "No sorry for…"
- NO_FAMILY_OUTREACH [NEW in 2.1]
  Definition: Faults the speaker for not reaching the bereaved families personally or first
  Example: "Did you go to their wake?" "Press first or the families first?" "Not even man enough to call their families"
- OTHER: Name it in the evidence field

Rule: use DISMISSIVE only when no other reason is given. "Sorry is not enough, go to jail" = WORDS_NOT_ENOUGH, not DISMISSIVE.

## SECTION I. FLAGS

I1. TARGET_OF_COMMENT (choose ONE)

- ORGANIZATION
- PERSON
- VICTIMS_FAMILY
- OTHER_COMMENTERS
- MEDIA
- NONE

I2. META_DISCOURSE [NEW] (0/1)
Comments about the online reaction itself: bashing, netizens, "apologists," defending the right to criticize.

- Include: "Sana tigilan na pambabash," "daming Ateneo apologist dito," "I'm not bashing, I'm outraged."

I3. SARCASM (0/1)
Literal meaning differs from intended meaning.

I4. UNCODABLE (0/1)
Spam, off-topic, or unintelligible. If 1, set all other codes to 0 / NONE / NA. Emoji-only comments with a clear emotion are not uncodable (see rule 9).

## SECTION J. RESIDUAL CATEGORIES

J1. EMERGENT (pilot mode only, free text)
Anything meaningful the codes above do not capture. Short label + one-line explanation.

J2. OTHER_NOTE (full-coding mode, free text)
Residual for content the frozen codebook misses. Report the % of comments with a non-empty OTHER_NOTE.

## DERIVED VARIABLES (COMPUTED IN ANALYSIS, NOT CODED)

- Negative other-condemning emotion
  Rule: ANGER OR CONTEMPT OR DISGUST
  Source: Haidt (2003)
- Strict moral outrage
  Rule: NORM_VIOLATION AND (ATTRIBUTION = INTENTIONAL OR COVER_UP) AND INJUSTICE AND SELF_INTEREST
  Source: Coombs & Tachkova (2024); Antonetti & Maklan (2016)
- Knowing-risk outrage [NEW] [ADDED]
  Rule: NORM_VIOLATION AND (IGNORED_WARNING OR RECURRENCE) AND negative emotion
  Source: Study extension
- Negligence-based outrage [ADDED]
  Rule: NORM_VIOLATION AND ATTRIBUTION = NEGLIGENCE AND negative emotion
  Source: Study extension
- Broad outrage
  Rule: negative emotion AND (BLAME_TARGETS ≠ NONE OR ATTRIBUTION = ACCUSATION OR ACTIONS includes PUNITIVE or RETALIATION)
  Source: Brady et al. (2023), DOC

## MAPPING TO THEORIES

- T1 Class and identity talk shape emotion: CLASS_FRAME, CLASS_DIRECTION, IDENTITY_FRAME, LANGUAGE (register) × emotions / outrage / actions
- T2 Emotions reveal (violated) values: VALUES × emotions; VALUES × NORM_VIOLATION / BLAME_TARGETS (value incongruence, P1)
- T3 Emotions linked to action: Emotions / outrage × ACTIONS (TPP, P3; compassion → punishment, P5; RETALIATION vs PUNITIVE)
- Cross-cutting: Video type; STANCE; RESPONSE_EVAL + REJECT_REASONS (P6–P10); META_DISCOURSE

## CHANGE LOG

- v2.2: added B9. JOY and B10. SURPRISE (completing Ekman's six basic emotions alongside the existing ANGER, DISGUST, SADNESS and FEAR); updated theoretical-basis citation. Output format unchanged apart from the two new binary codes.
- v2.1 (after v2 recode of pilot 01): added REJECT_REASONS values DISMISSIVE and NO_FAMILY_OUTREACH, which together absorb 8 of the 12 "OTHER" reject reasons in the v2 recode; added DISMISSIVE decision rule. No other codes changed; output format unchanged.
- v2.0 (after pilot pass 01): added EMOJI language; SYMPATHY_OFFENDER; ATTRIBUTION = ACCUSATION; IGNORED_WARNING; RECURRENCE; ALT_ACCOUNT; COACHING_STAFF blame target; SANCTITY_OF_LIFE value; IDENTITY_FRAME; APPEARANCE_ATTACK; actions RETALIATION, RESTITUTION, INFORMATION; REJECT_REASONS; META_DISCOURSE; NEEDS_CONTEXT; rules for replies, emojis, anonymization. Renamed COMPASSION → COMPASSION_VICTIMS. Added knowing-risk outrage.
- v1.0 — initial draft.
