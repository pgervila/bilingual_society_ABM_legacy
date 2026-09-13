# `city_objects.py` — city objects

Physical/institutional objects agents inhabit. ~989 lines. Several classes define
`__getstate__`/`__setstate__` for dill pickling.

## `Home` (`city_objects.py:11`)

Holds occupants. `assign_to_agent(agents)` places one agent or a whole family and
sets their `loc_info['home']`. `remove_agent(agent, replace=False, grown_agent=None)`
handles departure and (on `evolve`) swapping an aged agent instance for its successor.

## `EducationCenter` (base, `:71`) → `School` / `Faculty`

Manages **courses keyed by student age**. State lives in `grouped_studs`
(`{age_key: {'students': set, 'teacher': agent}}`) and `info` (`age_range`,
`students`, `lang_policy`).

### Course lifecycle (rolled over yearly by `model.update_centers`)

**`group_students_per_year()`** (`:85`) — bucket enrolled students into per-age courses.

**`set_up_courses()`** (`:97`) — initial setup: group students + hire teachers.

**`update_courses_phase_1(max_course_dist=3)`** (`:108`):
- Promote every course's students to `age_key + 1`; students past `age_range[1]` **exit** the school (`exit_studs` → some to university, rest to the job market).
- Reconcile teachers: build the set of teachers left with no course (`jobless`) and courses left with no teacher (`missing`), then **reassign** jobless teachers to nearby missing courses (within `max_course_dist` years). Remaining jobless teachers have their course reference cleared.
- Reassign course ids to each student's `loc_info`.

**`update_courses_phase_2()`** (`:181`):
- `hire_teachers([c_id])` for any course still without a teacher.
- Any teacher at/over `Teacher.age_high` retirement age → `evolve(Pensioner)`.

### Teacher acquisition (`find_teachers`, `hire_teachers`, `:196`/`:271`)
Teachers are sourced in priority order:
1. `get_free_staff_from_cluster` — unoccupied eligible adults in the same cluster.
2. `get_free_staff_from_other_clusters` — same, other clusters (agent relocates).
3. `get_employees_from_companies(num_teachers)` — convert an `Adult` working at a `Job` into a `Teacher` (`evolve`), respecting the school language policy.

`check_teacher_old_job` releases the teacher's previous job seat.

### Student/teacher membership
`assign_student`, `remove_student`, `remove_student_from_course`,
`remove_teacher_from_course`, `remove_course`, `assign_teacher`. `__getitem__(key)`
indexes `grouped_studs` by age key.

## `School(EducationCenter)` (`:473`)

`age_range=(1, 18)`. Adds `swap_teachers_courses()` (`:585`, called every 4 years by
`update_centers`) and school-specific staff sourcing/exit logic. `remove_employee`
handles teacher departure with optional backfill.

## `Faculty(EducationCenter)` (`:631`) and `University` (`:755`)

A `University` (`age_range=(19,23)`) holds multiple `Faculty` objects keyed by type
(`__getitem__(fac_key)`). Faculties behave like schools but for university-age
students (`YoungUniv`) and `TeacherUniv` staff. Only ~20% of towns get a university
(`map_universities(pct_univ_towns=0.2)`).

## `Job` (`:787`)

Workplace with `num_places` seats, `skill_level`, and a `lang_policy`. State in
`info` (`employees` set, `lang_policy`, `skill_level`, `clust`) and `agents_in`
(agents physically present). `lang_policy` values: `0` → only L1 required (agents of
type 0 or 1 may work), `1` → both languages (only bilingual type-1 agents), `2` →
only L2 (types 1 or 2).

- **`set_lang_policy(min_pct=0.1)`** (`:811`): derive the required languages from the **cluster's language distribution** (`geo.get_lang_distrib_per_clust`). A monolingual group counts only if it exceeds `min_pct` (10%) of the cluster. The rules collapse mixed cases toward requiring bilingualism: if `1` isn't already implied and there's fewer than two dominant groups, `1` is inserted; if both monolingual sides dominate (or all three groups qualify), the policy becomes `[1]` (bilingual-only). Refreshed yearly by `update_centers`.
- **`check_cand_conds(agent, keep_cluster=False, job_steps_years=3)`** (`:832`): eligibility check. Candidate must be a `Young` (excluding `Teacher`/`Pensioner`), either unemployed or past a seniority threshold (`job_steps_years` years normally, 1 year if `keep_cluster`), and — for cross-cluster moves — have a consort who is also free to move (and not a teacher).
- **`look_for_employee(excluded_ag=None, all_clusters=False)`** (`:869`): scan the cluster's agents (or all agents) for a `Young` non-`Pensioner` passing `check_cand_conds` and not temporarily `blocked`, then hire the first match.
- **`hire_employee(agent, move_home=True, ignore_lang_constraint=False)`** (`:897`): sets a temporary `agent.blocked` flag (prevents re-hiring mid-cascade), removes the agent from any old job, then hires **only if** `agent.info['language'] in lang_policy` (or `ignore_lang_constraint`). Relocates the agent's home if the job is in another cluster. If hiring fails, it's necessarily a language mismatch → `agent.react_to_lang_exclusion(lang)`.
- **`assign_employee(agent)`** (`:939`): add to `employees`, set `agent.loc_info['job']`, reset `job_steps`. (Seat-count decrement is currently disabled pending an endogenous-economy model.)
- **`remove_employee(agent, replace=None, new_agent=None)`** (`:953`): free the seat; either install `new_agent` directly or (outside `init_mode`) `look_for_employee` to backfill.

`__getstate__`/`__setstate__` convert the `employees`/`agents_in` sets to tuples for
dill pickling.

## `MeetingPoint` / `Store` / `BookStore` (`:982`-`:989`)

Minimal stubs — placeholders for social/commercial venues, largely unused in the
current step logic.
