# Design QA — PKB Agent Workbench

## Visual truth and capture

- Source visual truth: `/var/folders/pr/3m4lhl6d2sdfk5syx9x0h1j80000gn/T/codex-clipboard-9c3df547-23b0-4499-a593-59f8461d0a15.png` (2048 × 1152).
- Implementation desktop capture: in-app Browser at `http://localhost:3001/` (2048 × 1152 CSS viewport); source and implementation were emitted together for side-by-side visual comparison in this task.
- Responsive implementation capture: in-app Browser at `http://localhost:3001/` (390 × 844 CSS viewport).

## Comparison passes

### Desktop, reference-matched conversation state

- **Layout and spacing:** matched the reference’s fixed pale sidebar, centered two-way segmented control, large intentional blank space, and single wide floating composer. The sidebar is capped at 356px at the reference width so the frame division remains visually comparable.
- **Typography and hierarchy:** used a neutral system sans stack, strong greeting headline, quiet supporting copy, and a restrained small-text hierarchy. The PKB-specific labels replace ChatGPT product copy without changing the visual rhythm.
- **Color, surfaces, and elevation:** white canvas, `#f7f7f8` sidebar, soft neutral active fill, hairline borders, and a low-elevation composer preserve the reference’s calm, non-dashboard surface language. There are no gradients or decorative illustrations.
- **Icons and assets:** visible controls use the Phosphor icon library with a consistent thin-stroke family. No custom SVG/CSS/emoji art is used in place of a source asset.
- **Copy and empty state:** the welcome copy explains the document → evidence-backed answer workflow; the empty knowledge-base status gives a clear next action.

### Responsive and interaction pass

- At 390 × 844, the sidebar collapses to a compact header, the segmented control stays tappable, the composer remains fully visible, and the suggestion list wraps into a readable single column without horizontal overflow in the rendered capture.
- Tested: switching between **对话** and **资料**, opening the knowledge-base empty state, clicking a prompt suggestion to populate the composer, and resetting that draft through **新对话**. No model request was sent during QA.
- Accessibility checks: semantic navigation/tablist/tab controls and labelled composer/upload controls are present; the disabled send state is exposed; visible focus outlines and a reduced-motion mode are implemented in `web/app/globals.css`.

### Final comparison adjustments

1. **P2 — Sidebar active fill drift.** The conversation navigation row had a filled active background that made the sidebar denser than the source. Removed that extra fill while retaining the explicit new-chat active state.
2. **P2 — Welcome vertical rhythm.** The greeting/composer group sat slightly low relative to the reference. Tightened the welcome offset to restore the intended visual center.

## Verification

- `npm run lint` — passed.
- `npm run build` — passed (Next.js production build and TypeScript).
- `npm audit --omit=dev` — passed, 0 production vulnerabilities.
- `git diff --check` — passed.
- Browser console errors at the verified preview — none.

final result: passed
