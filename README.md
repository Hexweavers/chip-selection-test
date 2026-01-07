We are building WinSpeak, a Duolingo-like platform to help users train their communication skills and improve their job interview performance.
Instead of generating exercises on-demand per user, we pre-generate a large, reusable pool of validated exercises.
When a user starts a session, we intelligently select exercises that match their curriculum goals, personal context (industry/role/chips), current skill levels, and ensure novelty.
The full system generates, organizes, and delivers personalized interview practice exercises to WinSpeak users.
It combines AI generation, intelligent selection algorithms, automated scoring, and text-to-speech to create an adaptive learning experience.

Chips are curated, structured tags representing situations, jargon, or domain-specific concepts relevant to a user's work context. Example of one chip:

Key: "stakeholder_conflict"
Display: "Stakeholder Conflict"
Types: ["situation"]
Synonyms: ["competing_priorities", "alignment_challenges"]
Related: ["negotiation", "consensus_building"]

**Chip Types:**
- **Situation:** Workplace scenarios (e.g., `stakeholder_conflict`, `tight_deadline`, `ambiguous_requirements`)
- **Jargon:** Industry/role-specific terms (e.g., `okrs`, `sprint_planning`, `p_and_l`)
- **Role Task:** Common responsibilities (e.g., `roadmap_prioritization`, `customer_discovery`)
- **Environment:** Organizational contexts (e.g., `remote_team`, `startup_environment`, `enterprise_processes`)
 
In this project/POC right here, I want to create a test for our platform of how chip generation works between different models and with different prompts. This is relevant for our sign-up process, where we'll be collecting relevant information from users to generate Chips.

In the sign-up process, the user will select one of 21 different Sectors:

Technology & Software
Healthcare & Medical
Finance & Accounting
Sales & Business Development
Customer Service & Call Centers
Retail & Merchandising
Hospitality & Food Service
Manufacturing, Engineering & Industrial
Construction, Real Estate & Property Services
Education & Training
Government & Public Sector
Legal & Compliance
Pharmaceutical & Biotechnology
Media, Marketing & Communications
Energy & Utilities
Logistics, Transportation & Warehousing
Professional Services (Consulting, Accounting, Legal)
Nonprofit & Social Services
Science & Research (R&D, Labs)
Creative & Design (UX, Graphic, Content)
Other

After choosing a Sector, the user will type their Desired Role into a text box, e.g. "Senior Software Engineer", "Chief Nurse", "Accounting Manager", "Head of Logistics", "Product Owner", "Real Estate Agent", etc. 

So, here's where we come in. After they insert their sector + desired_role, we can now quickly generate 8-10 chips (using a fast LLM) that will be shown to the user, and they can pick these to specify their role a bit more. 
Or we can just end the sign-up right there (making the flow fast and practical) and use just their sector + desired_role to generate chips.

Does this make sense? Can you help me brainstorm how we can test this out?

