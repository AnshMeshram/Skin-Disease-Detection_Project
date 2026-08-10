# Figma Design Prompt — SkinAI Skin Disease Detection

## Use this prompt in Figma AI / with a designer

---

### Overview
Design a dark, clinical-luxe medical AI web application for dermoscopy skin disease detection. The interface should feel like premium medical software — precise, trustworthy, and technically sophisticated.

---

### Color Palette
```
Background Primary:   #04101E  (deep navy)
Background Card:      #081828  (slightly lighter navy)
Accent Blue:          #0EA5E9  (electric sky blue)
Accent Cyan:          #22D3EE  (lighter cyan)
Blue Dim (fills):     rgba(14,165,233,0.10)
Blue Border:          rgba(14,165,233,0.20)
Text Primary:         #EFF6FF  (near-white blue tint)
Text Secondary:       #94A3B8  (slate gray)
Text Muted:           #475569  (dim slate)
Success:              #10B981
Warning:              #F59E0B
Danger:               #F43F5E
```

---

### Typography
```
Display/Headings: Syne — weight 700–800, tight tracking (-0.035em)
Body/UI:          DM Sans — weight 300–500
Font scale:
  H1:   clamp(3rem, 6.5vw, 5rem) / Syne 800
  H2:   2rem / Syne 700
  H3:   1rem / Syne 600
  Body: 1rem / DM Sans 400
  Small: 0.8rem / DM Sans 400
  Micro: 0.72rem / DM Sans 500
```

---

### Component Specs

#### Navigation Bar
- Fixed, 100% width
- Background: rgba(4,16,30,0.85) with backdrop blur 16px
- Bottom border: 1px solid rgba(14,165,233,0.20)
- Left: Logo "Skin**AI**" — "AI" in electric blue #0EA5E9
- Center: Pill tab switcher (Home | Model | Students)
  - Pill container: rgba(14,165,233,0.10) bg, 1px blue border, 999px radius
  - Inactive tab: slate text, transparent bg
  - Active tab: #0EA5E9 bg, white text
- Right: Small API status indicator (dot + "API Online" text)

#### Hero Section
- Full viewport height (100vh)
- Background: blurred medical photo (stethoscope, pills, clipboard on blue) with brightness 0.25
- Overlay gradient: transparent → rgba(4,16,30,0.7) → solid #04101E at bottom
- Grid overlay: subtle 60px grid lines at 4% opacity blue

**Hero content (centered, max-width 680px):**
1. Animated badge pill: "● AI · EfficientNet-B3 + InceptionV3 + ConvNeXt Ensemble"
   - Cyan text, blue border, dot pulses
2. H1: "Detect **Disease** Instantly" — "Disease" in #0EA5E9
3. Subtitle: 2-line descriptor, slate gray, max-width 480px
4. Camera capture box (360×225px):
   - Dark navy fill with 4% blue tint
   - 1.5px dashed blue border, 20px radius
   - Four corner bracket decorations in electric blue (viewfinder style)
   - Center: camera icon (stroke, 42px) + "Click to upload image" + sub-text
   - On hover: border brightens, bg darkens slightly
   - On image loaded: image fills box, animated scan line sweeps top-to-bottom
5. Two buttons side by side:
   - Primary: "Analyze Image" — solid #0EA5E9, white text, pill shape
   - Secondary: "Upload Image" — transparent, blue border, slate text

**Model badges (below hero content, flex row):**
- Three cards with glassmorphism (rgba dark + backdrop blur)
- Each: icon circle (blue-dim fill) + model name (Syne) + "About the algorithm ↗" link
- Models: EfficientNet-B3 | Inception V3 | ConvNeXt Tiny

#### Preprocessing Pipeline Section
- Section label badge + H2 "6-Step Image Enhancement"
- Six thumbnail cards in a row with arrow connectors (→)
- Each card: square, dark navy bg, blue border, 12px radius
  - Placeholder icon (gray) before analysis
  - Processed image fills card after analysis
  - Active step: bright blue border + subtle glow
  - Done step: green-tinted border
- Labels below each card: name (small bold) + description (micro gray)
- Step names: White Balance → Original → Hair Removed → ITA CLAHE → Skin Mask → Lesion Region

#### Results Card
- Background: rgba(8,24,40,0.85), blue border, 20px radius, backdrop blur
- Header row: "Analysis Results" badge + "New Analysis" ghost button
- Two-column grid (200px image | flex-1 info):
  - Image column: square, border, rounded, "Processed" badge overlay
  - Info column (top to bottom):
    - "Disease Detected" label + large Syne disease name in electric blue
    - Confidence: value % + animated gradient bar (blue → cyan)
    - Symptoms text row
    - Divider line
    - Model used (as chip/badge)
    - Ensemble vote
- Warning disclaimer bar (amber tint, amber border, warning icon)

#### Model Architecture Section
- Three-way pill tab switcher (EfficientNet-B3 | Inception V3 | ConvNeXt Tiny)
- Per tab:
  - 4-column stat grid (dark card each: value in blue Syne, label in gray)
  - Detail card: H3 + paragraph + tag row (small cyan chips)

#### Disease Class Grid
- 9 cards in auto-fill grid (min 180px)
- Each: 2-digit number in large faded blue + name + abbreviation

#### Team Section
- 4-column card grid
- Each card: avatar circle (initials in blue) + name (Syne) + role (gray)
- Hover: lifts with subtle blue glow

---

### Micro-interactions & Motion
- Page load: staggered fade-up for hero elements (badge → H1 → desc → camera → buttons → badges)
- Camera scan line: continuous sweep animation when image loaded
- Pipeline steps: sequential activate (200ms apart) during analysis
- Confidence bar: width animates 0% → value over 1.2s cubic-bezier
- Results card: fade-up animation on appear
- Model tab switch: slide-in from left on panel change
- Scroll reveal: sections fade-up as they enter viewport
- Nav: compresses slightly on scroll
- Cards: translateY(-2px) on hover with border brightening

---

### Layout / Spacing
- Max content width: 1000px centered
- Section padding: 5rem vertical, 2rem horizontal
- Card gap: 1rem standard, 2rem for major grids
- Border radius: 14px standard, 20px for large cards
- All borders: 1px solid rgba(14,165,233,0.20)

---

### Asset Requirements
- hero-bg.jpg: medical flatlay (stethoscope, clipboard, pills) on bright blue background
  - Will be blurred and darkened via CSS filter
  - Suggested Unsplash query: "medical stethoscope blue background flat lay"
- No other images required (pipeline images come from backend API)
