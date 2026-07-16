// In-app documentation content (Help). Plain static data, rendered by
// routes/help/+page.svelte with {@html}. No user input ever flows into these
// strings, so this is not an XSS surface — keep it that way.

export interface DocSection {
  id: string;
  group: string;
  title: string;
  /** Extra words the search box should match, beyond title + body text. */
  keywords?: string[];
  html: string;
}

export const docGroups: string[] = [
  'Start here',
  'Importing your photos',
  'Alignment',
  'Stacking methods',
  'Large & batch stacks',
  'Running jobs',
  'Viewing & comparing',
  'Retouching',
  'Exporting',
  'RAW & Capture One',
  'Projects & file safety',
  'Troubleshooting',
  'Reference'
];

export const docSections: DocSection[] = [
  // ───────────────────────── Start here ─────────────────────────
  {
    id: 'welcome',
    group: 'Start here',
    title: 'Welcome to PlaneFuse',
    keywords: ['about', 'overview', 'intro'],
    html: `
      <p>PlaneFuse merges a sequence of photos taken at different focus distances
      into a single image with more sharp detail than any one frame could hold on
      its own — a technique called <strong>focus stacking</strong>. It runs
      entirely on your own computer: nothing is uploaded anywhere, and it works
      with plain developed images (JPEG/TIFF/PNG) or, for macro and product
      photographers who want to keep grading control in another app, directly
      with camera RAW files.</p>
      <p>This guide is written for photographers, not programmers. It explains
      both <em>what to click</em> and <em>why it works that way</em>, so you can
      make good decisions even on stacks that don't behave like the examples.
      Every screen in the app links back here — press <kbd>?</kbd> anywhere in
      the workspace for a quick shortcut list, or use the <strong>Help</strong>
      link in the top bar to come back to this page.</p>
      <div class="callout tip">
        <strong>New here?</strong> Read <a href="#what-is-focus-stacking">What is
        focus stacking?</a>, then follow <a href="#first-stack-walkthrough">Your
        first stack: a walkthrough</a> with a real folder of photos open in the
        other window.
      </div>`
  },
  {
    id: 'what-is-focus-stacking',
    group: 'Start here',
    title: 'What is focus stacking?',
    keywords: ['depth of field', 'macro', 'concept', 'basics'],
    html: `
      <p>Every lens has a limited <strong>depth of field</strong>: the range of
      distances that look acceptably sharp at a given aperture. Get close to a
      subject — macro work, product shots, anything shot wide open for a soft
      background — and that sharp range can shrink to a few millimetres.
      Stopping down the aperture buys you more depth of field, but past a certain
      point diffraction softens the whole image, and very small apertures can
      still fall short on genuinely deep subjects (a full flower, an insect head
      to abdomen, a watch face at high magnification).</p>
      <p>Focus stacking sidesteps the trade-off. Instead of one exposure, you
      shoot a <strong>sequence</strong> where the focus point steps through the
      subject from front to back (or back to front) at a wide-open or
      moderate aperture, where the lens is optically at its best. Each frame is
      sharp somewhere different. PlaneFuse then:</p>
      <ol>
        <li><strong>Aligns</strong> the frames, because focus-racking a lens
        subtly shifts and rescales the image (focus breathing), and handheld or
        rail-based sequences drift a little from frame to frame.</li>
        <li><strong>Measures local sharpness</strong> in every frame, pixel
        region by pixel region.</li>
        <li><strong>Combines</strong> the sharpest information from every frame
        into one image, using whichever <a href="#choosing-a-method">stacking
        method</a> suits your subject.</li>
      </ol>
      <p>The result is a single photo that is sharp across the whole stacked
      range — something no single exposure of that scene could have produced.</p>`
  },
  {
    id: 'shooting-a-stack',
    group: 'Start here',
    title: 'Shooting a stack that stacks well',
    keywords: ['capture', 'technique', 'tripod', 'rail', 'overlap', 'aperture', 'lighting'],
    html: `
      <p>Good software cannot fully rescue a poorly shot sequence. A few habits
      make an enormous difference to the final result:</p>
      <ul>
        <li><strong>Lock exposure and white balance.</strong> Shoot manual
        exposure (or at least lock it) so brightness doesn't pulse frame to
        frame. PlaneFuse can optionally match brightness during alignment, but
        it works best correcting small drifts, not full auto-exposure swings.</li>
        <li><strong>Keep the frame-to-frame focus step small enough to
        overlap.</strong> Each step's sharp zone should overlap the next one's.
        Too big a step leaves a soft band nothing can cover; too small just
        means more frames (and more time) than necessary.</li>
        <li><strong>Stabilise the camera.</strong> A tripod, focusing rail, or
        focus-bracketing feature on the camera all work. Handheld stacks are
        possible — alignment can absorb small shifts and rotations — but
        every extra millimetre of drift is a millimetre alignment has to
        correct instead of you.</li>
        <li><strong>Keep lighting consistent across the sequence.</strong>
        Continuous or well-recycled flash lighting avoids exposure flicker that
        alignment and stacking cannot distinguish from focus differences.</li>
        <li><strong>Avoid anything that moves between frames</strong> — wind on
        a flower, a live insect, a shifting reflection. Moving elements can't be
        stacked correctly because they aren't in the same place in every frame;
        expect a ghost or a smeared region wherever that happened.</li>
        <li><strong>Shoot a few extra frames at both ends.</strong> It's cheap
        insurance against a soft foreground or background edge, and PlaneFuse's
        <a href="#smart-selection">smart frame selection</a> will simply mark
        genuinely redundant frames rather than force you to use every one.</li>
      </ul>
      <div class="callout note">
        <strong>RAW shooters:</strong> if you plan to grade the final image in
        Capture One afterwards, keep every frame in the sequence on the
        <em>same camera body, same lens/sensor mode, same crop</em> — see
        <a href="#same-camera-rule">the same-camera rule</a> before you start
        shooting a mixed-gear sequence you can't later merge.
      </div>`
  },
  {
    id: 'first-stack-walkthrough',
    group: 'Start here',
    title: 'Your first stack: a walkthrough',
    keywords: ['tutorial', 'getting started', 'step by step', 'quick start'],
    html: `
      <p>This is the shortest path from a folder of photos to an exported,
      sharp image.</p>
      <ol>
        <li><strong>Open PlaneFuse.</strong> The workspace opens with an
        untitled scratch project — you don't have to create anything first.</li>
        <li><strong>Add your frames.</strong> In the left <em>Inputs</em> panel,
        click <strong>+ Add…</strong>, then <strong>📁 Pick folder…</strong> (or
        <strong>🖼 Pick files…</strong>) to open a native file dialog. Your
        photos are read from where they already live — nothing is copied. See
        <a href="#adding-frames">Adding photos to a project</a>.</li>
        <li><strong>Check the validation panel.</strong> Below the input list,
        PlaneFuse tells you whether the frames are compatible (matching size
        and bit depth, or matching camera for RAW) and names any file that
        isn't. The <strong>Run</strong> button stays disabled until this is
        clean — see <a href="#validation-panel">Reading the validation panel</a>.</li>
        <li><strong>Pick a stacking method.</strong> In the toolbar, click one
        of the method buttons — <code>pmax</code> is the sensible default for a
        first try. See <a href="#choosing-a-method">Choosing a stacking
        method</a> if you want to know when to reach for the others.</li>
        <li><strong>Leave "Align frames" checked</strong> unless your sequence
        came off a telecentric/rail setup that is already pixel-registered.
        Alignment is on by default and handles normal focus breathing and small
        drift automatically.</li>
        <li><strong>Click Run.</strong> The job appears in the render drawer at
        the bottom of the window with a live progress bar. When it finishes, the
        result appears as a thumbnail and opens automatically in the viewer.</li>
        <li><strong>Look it over.</strong> Zoom in (press <kbd>Z</kbd> for 100%,
        <kbd>F</kbd> to fit), switch to <strong>Result / source</strong> compare
        mode to check a specific region against a single input frame, and open
        the histogram to check for clipping. See <a href="#viewer-basics">The
        viewer</a>.</li>
        <li><strong>Export.</strong> Click <strong>Export</strong> in the
        toolbar, choose a format and destination folder, and click
        <strong>Export</strong> in the panel. See <a href="#export-formats">Export
        formats & bit depth</a>.</li>
      </ol>
      <div class="callout tip">
        Not happy with the result? Nothing here is destructive — the original
        frames are untouched. Try a different method, or open <strong>Options</strong>
        in the toolbar and tune the algorithm's parameters, then <strong>Run</strong>
        again. Multiple results sit side by side in the render drawer so you can
        compare them directly.
      </div>`
  },

  // ───────────────────────── Importing ─────────────────────────
  {
    id: 'adding-frames',
    group: 'Importing your photos',
    title: 'Adding photos to a project',
    keywords: ['import', 'folder', 'files', 'input list', 'scan'],
    html: `
      <p>Use the <strong>+ Add…</strong> button at the top of the Inputs panel.
      You have three ways to point PlaneFuse at your photos:</p>
      <ul>
        <li><strong>📁 Pick folder…</strong> opens your OS's native folder
        picker and adds every supported image directly inside it.</li>
        <li><strong>🖼 Pick files…</strong> opens a native file picker for
        selecting individual images (handy for hand-picking frames out of a
        larger folder, or combining frames from two folders).</li>
        <li><strong>Paste a folder / file path</strong> into the text field —
        useful when you already have the path copied, or the app is running
        somewhere a native dialog can't reach (e.g. inside a container).</li>
      </ul>
      <p>Frames are always read <strong>in place</strong> from disk — PlaneFuse
      never copies or moves your originals into the project. You can add more
      frames later with the same button, and remove any single frame with the
      <strong>✕</strong> that appears when you hover its row.</p>
      <p>Each added frame gets a status: <code>ok</code> means it decoded and
      matches the rest of the stack; anything else blocks the run and is
      explained in the <a href="#validation-panel">validation panel</a> right
      below the list, with a link that jumps straight to the offending frame.</p>`
  },
  {
    id: 'rendered-vs-raw',
    group: 'Importing your photos',
    title: 'Rendered images vs. camera RAW',
    keywords: ['jpeg', 'tiff', 'png', 'domain', 'develop', 'raw workflow'],
    html: `
      <p>PlaneFuse works in one of two modes per stack, decided automatically
      by what you add — you never pick it manually:</p>
      <ul>
        <li><strong>Rendered / developed images</strong> — JPEG, TIFF, or PNG
        files that have already been through your camera's or a raw
        converter's development (white balance, tone curve, colour, etc.).
        This is the simplest path: shoot, develop each frame the same way (or
        stack straight out of camera JPEGs), stack, export a finished TIFF/PNG/JPEG.</li>
        <li><strong>Camera RAW, "no-bake" mode</strong> — add RAW files
        (the formats your installed decoder supports) and PlaneFuse decodes
        them itself using a fixed, minimal recipe: no white balance, no gamma,
        no auto brightness, no denoise, no sharpening, no colour-space
        conversion. The stack is fused in the camera's own linear light values
        and exported as a <a href="#linear-dng">Linear DNG</a> that a raw
        developer such as Capture One opens and grades exactly like any other
        RAW file — you keep full creative control downstream instead of
        inheriting PlaneFuse's opinion of white balance or contrast.</li>
      </ul>
      <p>You cannot mix the two domains in one stack — the validation panel
      will flag it as <em>mixed RAW and rendered files</em>. Pick one path per
      shoot: develop everything first if you want quick, finished output;
      shoot and stack RAW if you want to grade the merged result yourself
      afterwards.</p>
      <div class="callout note">
        The <strong>Tonemap</strong> button in the viewer only affects how a
        scene-linear RAW result <em>previews on screen</em> (a gentle
        exposure/gamma preview curve so it doesn't look flat and dark). It never
        touches the exported pixels — see
        <a href="#tonemap-toggle">The Tonemap toggle</a>.
      </div>`
  },
  {
    id: 'same-camera-rule',
    group: 'Importing your photos',
    title: 'The same-camera rule for RAW stacks',
    keywords: ['calibration', 'sensor mode', 'cfa', 'mixed camera'],
    html: `
      <p>A RAW file carries per-camera calibration data — colour matrices, black
      and white levels, the sensor's colour filter array pattern — that a raw
      developer needs to render color correctly. A stacked image only has one
      set of pixels but originally came from several exposures, so PlaneFuse
      can only honestly claim <em>one</em> calibration for the whole result.
      That means every frame in a RAW stack must share:</p>
      <ul>
        <li>the same camera body and sensor mode,</li>
        <li>the same active sensor dimensions, orientation, and colour filter
        array pattern,</li>
        <li>the same bit-depth/precision and calibration data.</li>
      </ul>
      <p>If any frame doesn't match, the validation panel names it specifically
      (<em>different camera</em>, <em>different sensor mode</em>, <em>different
      sensor pattern</em>, <em>different camera calibration</em>, <em>missing
      RAW calibration</em>) rather than silently guessing. This is a
      correctness rule, not an arbitrary restriction — mixing calibrations
      would produce a DNG whose embedded metadata lies about at least one of
      the source frames.</p>
      <p>In practice: shoot the whole sequence with one body without swapping
      lenses to something that changes the reported sensor mode, and you'll
      never see this warning.</p>`
  },
  {
    id: 'validation-panel',
    group: 'Importing your photos',
    title: 'Reading the validation panel',
    keywords: ['errors', 'run disabled', 'blocking', 'wrong size', 'bit depth'],
    html: `
      <p>The panel under the input list summarises whether the current set of
      frames can be stacked, and lists every frame that can't. <strong>Run</strong>
      in the toolbar stays disabled until it reads <em>Ready to stack</em>.</p>
      <table>
        <thead><tr><th>Message</th><th>What it means</th></tr></thead>
        <tbody>
          <tr><td>Different dimensions</td><td>This frame's pixel width/height
          doesn't match the rest of the stack.</td></tr>
          <tr><td>Different bit depth</td><td>Mixed 8-bit and 16-bit sources in
          a rendered stack.</td></tr>
          <tr><td>Mixed RAW and rendered files</td><td>The stack combines
          camera RAW with already-developed images — pick one
          <a href="#rendered-vs-raw">domain</a> per stack.</td></tr>
          <tr><td>Different camera / sensor mode / sensor pattern / camera
          calibration</td><td>See <a href="#same-camera-rule">the same-camera
          rule</a> for RAW stacks.</td></tr>
          <tr><td>Missing RAW calibration</td><td>The file's RAW metadata
          doesn't include the calibration PlaneFuse needs to keep the DNG
          honest.</td></tr>
          <tr><td>RAW could not be decoded</td><td>The decoder rejected the
          file — it may be corrupt, or an unsupported RAW variant.</td></tr>
          <tr><td>Unreadable file / Unsupported file</td><td>Not a recognised
          image, or the file couldn't be opened at all.</td></tr>
        </tbody>
      </table>
      <p>Click a blocker's file name to jump straight to its row in the input
      list, where you can remove it with the <strong>✕</strong> that appears on
      hover and try again without it.</p>
      <p>Stacks with more than 40 frames get an on-screen recommendation to run
      <a href="#smart-selection">smart frame selection</a> first — it isn't
      required, just a nudge, since large sequences often contain redundant
      frames that only add time without adding sharpness.</p>`
  },

  // ───────────────────────── Alignment ─────────────────────────
  {
    id: 'why-alignment',
    group: 'Alignment',
    title: 'Why alignment matters',
    keywords: ['focus breathing', 'registration', 'drift'],
    html: `
      <p>Racking a lens's focus subtly changes what the sensor sees — a
      phenomenon called <strong>focus breathing</strong> — shifting and
      slightly rescaling the frame even on a rock-solid tripod. Add any real
      handheld or rail drift and the same subject point can land on a
      different pixel in every frame. Stacking algorithms combine pixels
      <em>at the same coordinate</em> across frames, so without correcting for
      that drift first, the result would show doubled edges and misaligned
      detail wherever the drift was largest.</p>
      <p><strong>Align frames</strong> in the toolbar (on by default) estimates
      and corrects this drift automatically before stacking, using one frame
      as a fixed reference and warping every other frame onto it.</p>
      <div class="callout note">
        Only turn alignment off ("frames are pre-aligned") for sequences that
        are genuinely already pixel-registered — a telecentric lens setup or a
        rail that guarantees zero lateral drift. Turning it off on an ordinary
        sequence will not error, but it will leave any real drift in the final
        image as ghosting.
      </div>`
  },
  {
    id: 'alignment-models',
    group: 'Alignment',
    title: 'Alignment models: translation, similarity, perspective',
    keywords: ['warp', 'transform', 'rotation', 'scale'],
    html: `
      <p>Alignment picks a reference frame, then solves a geometric transform
      that maps every other frame onto it. The <strong>Alignment model</strong>
      dropdown (under "Align frames") controls how flexible that transform is
      allowed to be:</p>
      <ul>
        <li><strong>Translation</strong> — corrects sideways/up-down shift
        only. Fastest, and enough for a well-built rail with no rotation or
        scale change.</li>
        <li><strong>Similarity</strong> (default) — shift, rotation, and
        uniform scale. This is the right choice for most handheld or
        rail-plus-breathing sequences, since focus breathing mostly looks like
        a small, even zoom.</li>
        <li><strong>Perspective</strong> — a full projective transform, adding
        skew/keyed perspective correction on top of similarity. Use it if the
        camera or subject moved in a way that isn't a pure shift/rotate/zoom —
        for example a slight tilt between frames.</li>
      </ul>
      <p>Once a transform is estimated, frames are resampled with the chosen
      <strong>Warp interpolation</strong>: <strong>Lanczos-3</strong> (default,
      sharpest, slower), <strong>Bicubic</strong>, or <strong>Bilinear</strong>
      (fastest, softest). Lanczos-3 is the right choice unless you're
      iterating quickly on parameters and want faster turnarounds while you
      dial things in.</p>
      <p>Alignment first downsamples frames to <strong>max long edge</strong>
      (default 2048px) to estimate the transform quickly, then applies the
      resulting transform at full resolution — the estimate doesn't need
      full-resolution pixels to be accurate, and this keeps large stacks fast.</p>`
  },
  {
    id: 'alignment-options',
    group: 'Alignment',
    title: 'Brightness matching and quality exclusion',
    keywords: ['normalize brightness', 'correlation threshold', 'ecc', 'drop misaligned'],
    html: `
      <p>Two more toggles live under "Align frames":</p>
      <ul>
        <li><strong>Match brightness</strong> — corrects small exposure drift
        between frames (e.g. from flicker or minor flash inconsistency) as
        part of alignment. It is not a substitute for locking exposure at
        capture time, but smooths out the small differences that do slip
        through.</li>
        <li><strong>Exclude low quality</strong> — when enabled, any frame
        whose alignment correlation score falls below the <strong>ECC ≥</strong>
        threshold (default 0.9) is dropped from the stack rather than fused in
        misaligned. Raise the threshold to be stricter, lower it if a
        legitimately difficult frame (fine texture, low contrast) is being
        excluded when you'd rather keep it.</li>
      </ul>
      <p>Whether or not you enable exclusion, every alignment result records a
      per-frame-pair quality score you can inspect afterwards — see
      <a href="#alignment-review">Reading the alignment review</a>.</p>`
  },
  {
    id: 'alignment-review',
    group: 'Alignment',
    title: 'Reading the alignment review',
    keywords: ['quality score', 'recovered', 'excluded', 'low-quality link'],
    html: `
      <p>After a stack finishes, an <strong>Alignment</strong> disclosure
      appears in the render drawer header when that result used alignment.
      Expand it to see:</p>
      <ul>
        <li>which <strong>model</strong> was used and which frame was the
        <strong>reference</strong> everything else was warped onto,</li>
        <li>any frames that were <strong>recovered</strong> (successfully
        aligned despite an initial difficulty) or <strong>excluded</strong>
        (dropped for low quality, if you enabled that option),</li>
        <li>a <strong>quality passed</strong> badge, or a list of specific
        low-quality frame pairs with their correlation score if any pair
        scored below 0.9.</li>
      </ul>
      <p>A low-quality pair doesn't necessarily mean the result is wrong — it's
      a flag to go look at that region of the image at full zoom before you
      trust it, particularly on frames with very little texture (smooth
      backgrounds, blown highlights) where alignment naturally has less to
      work with.</p>`
  },

  // ───────────────────────── Stacking methods ─────────────────────────
  {
    id: 'choosing-a-method',
    group: 'Stacking methods',
    title: 'Choosing a stacking method',
    keywords: ['algorithm', 'which method', 'pmax vs dmap'],
    html: `
      <p>The toolbar lets you toggle on one or more methods and run them all in
      one click — each produces its own separate result, so comparing methods
      costs nothing but time. If you're not sure which to use, this is the
      fastest way to find out.</p>
      <table>
        <thead><tr><th>Method</th><th>Best for</th><th>Trade-off</th></tr></thead>
        <tbody>
          <tr><td><a href="#method-pmax">PMax</a></td><td>General-purpose
          default; textured, detailed subjects (insects, textiles, product
          detail)</td><td>Can look slightly harder-edged / more contrasty than
          the source, matching what dedicated stacking tools like Zerene
          produce with their PMax method.</td></tr>
          <tr><td><a href="#method-dmap">DMap</a></td><td>Smooth subjects,
          scenes where a clean, natural-looking depth transition matters more
          than pixel-perfect micro-contrast</td><td>Can soften extremely fine
          detail relative to PMax; also the only method that can output a
          usable depth map.</td></tr>
          <tr><td><a href="#method-weighted">Weighted</a></td><td>Subjects with
          smooth focus falloff where you want a gentler, more blended
          transition and less halo risk</td><td>Softer overall than PMax at
          equivalent settings.</td></tr>
          <tr><td><a href="#method-slab">Slab</a></td><td>Very deep stacks
          (100+ frames) where doing one giant fusion pass is slow or
          memory-heavy</td><td>Adds one more layer of parameters (how frames
          are grouped into sub-stacks) to reason about.</td></tr>
        </tbody>
      </table>
      <p>If you only ever try one thing: leave <strong>PMax</strong> selected
      with its default parameters. It's the general-purpose choice and the one
      most comparable to what other stacking software calls "PMax".</p>`
  },
  {
    id: 'method-pmax',
    group: 'Stacking methods',
    title: 'PMax (pyramid max)',
    keywords: ['laplacian', 'halo', 'selection smoothing'],
    html: `
      <p>PMax decomposes every frame into a <strong>Laplacian pyramid</strong> —
      a multi-scale representation that separates fine detail from coarse
      structure — and at every scale, in every region, keeps whichever
      frame's detail has the most local energy (i.e. is locally sharpest). It
      then collapses the winning detail back into one image.</p>
      <p>Because it always picks the objectively sharpest information at every
      scale, PMax tends to look punchy and highly detailed — but it can also
      amplify contrast and noise a little more than a straight photo would,
      and can show faint halos at hard focus-transition edges. That's expected
      behaviour, not a bug: it's the same trade-off dedicated stacking tools
      make with their own PMax-style methods.</p>
      <p>Its one tunable parameter:</p>
      <ul>
        <li><strong>Selection smoothing</strong> (0–3, default 1) —
        median-filters the per-scale "which frame won this region" map before
        collapsing, which suppresses halo artefacts at the cost of a little
        speed. <code>0</code> disables it for the fastest run; raise it if you
        see visible seams at focus-transition edges.</li>
      </ul>`
  },
  {
    id: 'method-dmap',
    group: 'Stacking methods',
    title: 'DMap (depth map)',
    keywords: ['depth map', 'contrast threshold', 'smoothing radius'],
    html: `
      <p>DMap builds a single <strong>depth map</strong> first — for every
      pixel, which frame index was sharpest there — smooths that map so it
      transitions gradually rather than jumping frame-to-frame at every pixel,
      then blends the source frames using that smoothed map with fractional
      (in-between) weights. The result tends to look more natural and less
      contrasty than PMax, at some cost to the very finest detail.</p>
      <p>DMap is also the only method that can export its depth map as a
      companion file (a false-height map of the scene) — see
      <a href="#export-companions">export companions</a>.</p>
      <p>Parameters:</p>
      <ul>
        <li><strong>Estimation radius</strong> (2–40px, default 8) — size of
        the local window used to measure sharpness at each pixel. Larger
        values are more robust to noise but less precise about exactly where
        focus changes.</li>
        <li><strong>Contrast threshold</strong> (0–50%, default 7%) — pixels
        whose sharpness is below this percentile of the frame's maximum are
        treated as "undecided" (e.g. smooth, out-of-focus backgrounds with no
        texture to judge) and filled in from neighbouring decided regions
        instead of trusting a noisy sharpness comparison.</li>
        <li><strong>Smoothing radius</strong> (1–64px, default 16) — how far
        the edge-aware smoothing spreads when cleaning up the depth map before
        blending. Higher values give smoother transitions but can blur fine
        depth boundaries.</li>
      </ul>`
  },
  {
    id: 'method-weighted',
    group: 'Stacking methods',
    title: 'Weighted average',
    keywords: ['softmax', 'temperature', 'sharpness radius', 'blend'],
    html: `
      <p>Instead of picking one winning frame per pixel like PMax or DMap,
      Weighted blends <em>all</em> frames at every pixel, weighting each by how
      sharp it is there (a softmax over local sharpness). The result is the
      softest, most blended-looking of the three per-pixel methods, and the
      least prone to hard halos — a good option when a subject's focus falls
      off very gradually and you'd rather have a gentle blend than a crisp but
      occasionally seamy transition.</p>
      <p>Parameters:</p>
      <ul>
        <li><strong>Temperature</strong> (0.001–1.0, default 0.05) — lower
        values behave more like a hard "pick the sharpest frame" (closer to
        PMax's character, more halo-prone); higher values blend more frames
        together more evenly (softer, smoother, less detail).</li>
        <li><strong>Sharpness radius</strong> (2–40px, default 8) — size of the
        local window used to score each frame's sharpness, same idea as DMap's
        estimation radius.</li>
      </ul>`
  },
  {
    id: 'method-slab',
    group: 'Stacking methods',
    title: 'Slab (deep stacks)',
    keywords: ['hierarchical', 'sub-stack', 'inner method', 'outer method', 'overlap'],
    html: `
      <p>Slab is not a fourth way of comparing sharpness — it's a way of
      running one of the other three methods on very long sequences more
      efficiently. It splits the frame sequence into overlapping
      <strong>sub-stacks</strong> ("slabs"), fuses each slab with an
      <strong>inner method</strong>, then fuses the resulting slab outputs
      together with an <strong>outer method</strong>.</p>
      <p>Parameters:</p>
      <ul>
        <li><strong>Slab size</strong> (2–100, default 10) — frames per
        sub-stack.</li>
        <li><strong>Slab overlap</strong> (0–50, default 2) — how many frames
        consecutive slabs share, which keeps the transition between slabs
        smooth rather than seamed.</li>
        <li><strong>Inner method</strong> (default PMax) — the method used
        inside each slab.</li>
        <li><strong>Outer method</strong> (default DMap) — the method used to
        combine the slab results into the final image.</li>
      </ul>
      <p>Reach for Slab on sequences deep enough (100+ frames) that a single
      full-stack fusion pass is slow or memory-constrained on your machine.
      For typical stacks of a few dozen frames, PMax, DMap, or Weighted run
      directly on the whole sequence and Slab adds complexity without a real
      benefit.</p>`
  },
  {
    id: 'comparing-methods',
    group: 'Stacking methods',
    title: 'Running several methods at once',
    keywords: ['multi-select', 'compare results'],
    html: `
      <p>Click more than one method button in the toolbar (they toggle
      independently) and <strong>Run</strong> queues one job per method, each
      with its own default parameters. Every result lands in the render
      drawer as a separate thumbnail, so you can flip between them in the
      viewer and pick the one that looks best on your subject — no need to
      commit to a method before you've seen the results.</p>
      <div class="callout note">
        Per-method parameter tuning (the <strong>Options</strong> panel's
        sliders) only applies when exactly one method is selected. With
        several selected, each runs with its defaults; deselect down to one to
        tune it, run, then reselect others if you want a multi-way comparison
        with a tuned variant included.
      </div>`
  },
  {
    id: 'presets',
    group: 'Stacking methods',
    title: 'Presets',
    keywords: ['save settings', 'reuse parameters'],
    html: `
      <p>Open <strong>Options</strong> in the toolbar to see the
      <strong>Presets</strong> panel. It saves your current method selection,
      tuned parameters, and alignment/selection toggles under a name you
      choose, so a recipe that works for a particular subject (say, "macro
      insect, 40 frames") is one click away next time instead of re-entering
      every value.</p>
      <ul>
        <li><strong>Save current</strong> — type a name and click it to store
        the current settings.</li>
        <li>Click a saved preset's name to load it back into the toolbar.</li>
        <li>Click the <strong>✕</strong> next to a preset to delete it.</li>
      </ul>
      <p>Presets live with your PlaneFuse installation, not inside a single
      project, so they're available across every project you open.</p>`
  },

  // ───────────────────────── Large & batch stacks ─────────────────────────
  {
    id: 'smart-selection',
    group: 'Large & batch stacks',
    title: 'Smart frame selection',
    keywords: ['redundant frames', 'coverage', 'proposal', 'large stack'],
    html: `
      <p>Long sequences (especially ones shot with a small focus step, or
      bracketed generously "just in case") often contain frames that add
      almost nothing — their sharp region is already fully covered by
      neighbouring frames. Smart frame selection finds those and proposes
      dropping them, cutting stacking time without losing sharp coverage.</p>
      <p>To use it: enable <strong>Smart frame selection</strong> below the
      toolbar, then click <strong>Review proposal</strong>. PlaneFuse analyses
      the sequence and shows:</p>
      <ul>
        <li>how many frames are <strong>kept</strong> vs. flagged
        <strong>redundant</strong>,</li>
        <li>a <strong>coverage</strong> count (reliable cells / total cells) —
        how much of the frame area has confidently sharp coverage from the
        kept frames,</li>
        <li>a strip showing every frame, with redundant ones struck through,</li>
        <li>a warning if coverage looks incomplete even after the proposed
        selection.</li>
      </ul>
      <p>Nothing is dropped automatically — review the strip, and click
      <strong>Accept proposal</strong> to actually apply it. <strong>Run</strong>
      is disabled while selection is enabled until you've accepted a proposal,
      so you can never silently stack with an un-reviewed selection. Adding or
      removing frames after accepting resets the acceptance, since the
      proposal is no longer describing the current input set.</p>`
  },
  {
    id: 'auto-group',
    group: 'Large & batch stacks',
    title: 'Auto-group (batching several subjects)',
    keywords: ['batch', 'multiple stacks', 'stack all groups'],
    html: `
      <p>If a folder actually contains several separate focus sequences back
      to back (e.g. you photographed five specimens in one session, cycling
      focus for each), <strong>Auto-group</strong> in the toolbar detects the
      boundaries between sequences and proposes one group per subject.</p>
      <p>Click <strong>Auto-group</strong> to see the proposed groups — each
      listed with its frame count and the first/last file name in that group —
      then <strong>Stack all groups</strong> to queue one stacking job per
      group per selected method, all at once. Nothing is queued until you
      confirm; <strong>Cancel</strong> discards the proposal without running
      anything.</p>
      <p>This is a convenience for genuinely separate sequences, not a
      substitute for reviewing your input list — if it groups things
      differently than you intended, add/remove frames or re-run it after
      correcting the input set.</p>`
  },

  // ───────────────────────── Running jobs ─────────────────────────
  {
    id: 'job-queue',
    group: 'Running jobs',
    title: 'The render drawer & job queue',
    keywords: ['queue', 'pending', 'cancel', 'reorder', 'rerun', 'history'],
    html: `
      <p>The drawer at the bottom of the workspace (click <strong>Show/Hide</strong>
      to collapse it) has two things in it: finished <strong>results</strong>
      as thumbnails on the left, and the active/queued <strong>job
      queue</strong> alongside them.</p>
      <p>For any job you can see its type, live status and progress bar, and a
      short status message. Depending on its state you can:</p>
      <ul>
        <li><strong>Reorder</strong> a pending job earlier/later in the queue
        with the ← → buttons.</li>
        <li><strong>Cancel</strong> a pending or running job.</li>
        <li><strong>Run again</strong> a finished stacking job with the exact
        same parameters — handy after tweaking input frames, or just to
        reproduce a result.</li>
        <li>Expand <strong>Parameters</strong> to see the exact JSON parameters
        that job ran with — useful if you're trying to remember what produced
        a particular result, or reporting an issue.</li>
      </ul>
      <p>A failed job shows a plain-language error message rather than a raw
      crash trace, and stays visible in the queue (rather than disappearing)
      so you can see what was attempted.</p>
      <p>Every finished result also carries a <strong>Retouch</strong> button —
      see <a href="#retouch-open">Opening Retouch</a>.</p>`
  },
  {
    id: 'estimates',
    group: 'Running jobs',
    title: 'Time & memory estimates',
    keywords: ['badge', 'planning', 'how long'],
    html: `
      <p>Next to the alignment/selection toggles, a small badge (e.g.
      <code>≈ 45s · ≈ 2.1 GB</code>) previews roughly how long the currently
      selected method + settings will take and how much memory it will use, based
      on your frame count, resolution, and current compute device.</p>
      <div class="callout note">
        This is a planning estimate, not a benchmark promise — actual time
        depends on your specific hardware, other running load, and how
        cooperative the sequence is for alignment. Treat it as a ballpark for
        deciding whether to start a run now or step away for a coffee, not as
        a guaranteed number.
      </div>`
  },
  {
    id: 'device-performance',
    group: 'Running jobs',
    title: 'Compute device, GPU, and out-of-memory handling',
    keywords: ['cuda', 'mps', 'cpu', 'gpu', 'oom', 'tiled'],
    html: `
      <p>The top bar shows the compute device PlaneFuse is using
      (<strong>CUDA</strong> on an NVIDIA GPU, <strong>MPS</strong> on Apple
      silicon, or <strong>CPU</strong>) and free memory, so you always know
      what's actually running your stack. PlaneFuse auto-selects the best
      available device on launch; CPU always works, just more slowly on large
      stacks.</p>
      <p>If a stack is too large to fit in memory at once, PlaneFuse
      automatically retries with <strong>tiled processing</strong> (splitting
      the image into pieces and processing each separately), and falls back
      further to <strong>tiled CPU</strong> if the GPU still can't fit it. The
      job's message records which fallback path was taken, so a slow run isn't
      a silent mystery.</p>
      <p>If a run is uncomfortably slow: close other GPU-heavy applications,
      or lower the alignment proxy size (<strong>max long edge</strong> under
      "Align frames") — alignment doesn't need full resolution to estimate an
      accurate transform, and a smaller proxy is noticeably faster on large
      source images.</p>`
  },

  // ───────────────────────── Viewer ─────────────────────────
  {
    id: 'viewer-basics',
    group: 'Viewing & comparing',
    title: 'The viewer',
    keywords: ['deep zoom', 'pan', 'tiles', 'fit', '100%'],
    html: `
      <p>The main viewer streams your image in tiles, so even a very
      high-resolution stack result opens and zooms instantly instead of
      waiting to load a full-resolution file. Scroll to zoom (centred on your
      cursor), drag to pan.</p>
      <p>Keyboard: <kbd>F</kbd> fits the whole image to the window,
      <kbd>Z</kbd> jumps to 100% (one image pixel per screen pixel, centred on
      the current view). The bottom-left readout shows your current zoom
      percentage.</p>
      <p>Click any frame in the <em>Inputs</em> list, or any thumbnail in the
      render drawer, to load it into the viewer. Use the left/right arrow keys
      to step through input frames one at a time without touching the mouse —
      handy for scrubbing through a sequence to spot the frame a soft region
      actually came from.</p>`
  },
  {
    id: 'compare-modes',
    group: 'Viewing & comparing',
    title: 'Compare modes',
    keywords: ['split', 'side by side', 'before after', 'result source'],
    html: `
      <p>The floating control bar at the top of the viewer switches between
      four modes:</p>
      <ul>
        <li><strong>Single</strong> — just the current selection, full window.</li>
        <li><strong>Result / source</strong> — the current result on one side of
        a draggable divider, a single source frame on the other, sharing pan
        and zoom so you can drag the divider across a detail and see exactly
        what the stack changed versus one frame.</li>
        <li><strong>Result / result</strong> — compare two finished results
        side by side (choose the second one from the dropdown that appears) —
        the natural way to decide between two methods or two parameter sets.</li>
        <li><strong>Before / after</strong> — hold <kbd>\\</kbd> (or the on-screen
        button) to temporarily swap to the comparison image and see the
        difference at a glance; release to snap back.</li>
      </ul>
      <p>All panes share the same pan and zoom, so once you've framed a detail
      in one mode, switching modes or dragging the divider keeps you looking
      at the same spot.</p>
      <p>Press <kbd>Esc</kbd> to exit back to Single mode at any time.</p>`
  },
  {
    id: 'histogram',
    group: 'Viewing & comparing',
    title: 'Histogram & clipping',
    keywords: ['rgb', 'luminance', 'shadows', 'highlights'],
    html: `
      <p>Toggle <strong>Histogram</strong> in the viewer control bar to overlay
      a live RGB + luminance histogram of the currently displayed image,
      computed server-side from the actual pixel data (not a screen
      approximation). Below the curves, per-channel shadow/highlight clipping
      counts light up in that channel's colour whenever any pixels are clipped
      at that end — a quick check for blown highlights or crushed shadows
      before you export.</p>
      <p>For a scene-linear RAW result, the histogram reflects whatever the
      <a href="#tonemap-toggle">Tonemap toggle</a> is currently showing —
      toggle it to see the raw linear distribution versus the tonemapped
      preview.</p>`
  },
  {
    id: 'tonemap-toggle',
    group: 'Viewing & comparing',
    title: 'The Tonemap toggle (RAW preview only)',
    keywords: ['auto tonemap', 'preview', 'exposure', 'gamma', 'flat', 'dark'],
    html: `
      <p>Scene-linear camera RGB — the values a RAW stack is fused in — looks
      dark and flat on a normal screen, because it hasn't been through the
      gamma curve and exposure/contrast shaping a raw developer normally
      applies. The <strong>Tonemap</strong> button in the viewer control bar
      applies a gentle preview-only exposure, white balance, and gamma curve
      so a RAW result previews at a normal-looking brightness and contrast
      inside PlaneFuse.</p>
      <div class="callout tip">
        This toggle changes <strong>only what you see on screen</strong>. It
        never touches the pixels that get written to disk — the exported
        Linear DNG (and its optional float TIFF companion) always contain the
        original, untouched scene-linear values regardless of whether Tonemap
        is on or off when you export. Turn it off if you specifically want to
        judge the raw linear distribution, e.g. while reading the histogram
        for genuine sensor-level clipping.
      </div>
      <p>Your choice is remembered per project.</p>`
  },
  {
    id: 'viewer-shortcuts',
    group: 'Viewing & comparing',
    title: 'Viewer keyboard shortcuts',
    keywords: ['keys', 'hotkeys'],
    html: `
      <table>
        <thead><tr><th>Key</th><th>Action</th></tr></thead>
        <tbody>
          <tr><td><kbd>F</kbd></td><td>Fit image to window</td></tr>
          <tr><td><kbd>Z</kbd></td><td>Zoom to 100%</td></tr>
          <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>Step to the previous / next input frame</td></tr>
          <tr><td><kbd>\\</kbd> (hold)</td><td>Show the comparison image in Before/after mode</td></tr>
          <tr><td><kbd>Esc</kbd></td><td>Exit compare mode back to Single</td></tr>
          <tr><td><kbd>?</kbd></td><td>Toggle the on-screen shortcut cheat-sheet</td></tr>
        </tbody>
      </table>
      <p>Shortcuts are suppressed while a text field has focus, so typing a
      project name or export path never accidentally triggers a viewer
      action.</p>`
  },

  // ───────────────────────── Retouching ─────────────────────────
  {
    id: 'retouch-open',
    group: 'Retouching',
    title: 'Opening Retouch',
    keywords: ['manual fix', 'clone', 'paint from source'],
    html: `
      <p>No automatic stack is perfect everywhere — a moving element, a stray
      hair the aligner couldn't fully settle, or a region where one method
      simply chose a less flattering frame than another. <strong>Retouch</strong>
      lets you manually paint pixels from a different source or result into a
      copy of your chosen result, region by region.</p>
      <p>Open it from the <strong>Retouch</strong> button on any result's
      thumbnail in the render drawer. PlaneFuse creates (or reopens, if you've
      already started one) a retouch session for that result and switches to
      the Retouch screen.</p>`
  },
  {
    id: 'retouch-tools',
    group: 'Retouching',
    title: 'Retouch brush tools',
    keywords: ['radius', 'hardness', 'opacity', 'erase', 'undo', 'redo'],
    html: `
      <p>The left panel lists every available <strong>paint source</strong> —
      another result, or an original input frame — as a thumbnail. Select one,
      then paint directly on the canvas to bring that source's pixels into the
      working image wherever you paint.</p>
      <ul>
        <li><strong>Radius</strong>, <strong>Hardness</strong>, and
        <strong>Opacity</strong> sliders control the brush, the same way they
        would in any photo editor — a hard, low-opacity brush for precise
        touch-ups; a soft, full-opacity brush for broad blends.</li>
        <li><strong>Normal</strong> mode paints the selected source in;
        <strong>Erase</strong> mode paints the original working image back in,
        undoing a normal stroke without a separate undo step.</li>
        <li><kbd>[</kbd> / <kbd>]</kbd> shrink/grow the brush radius on the
        fly. Hold <kbd>Space</kbd> to pan instead of paint.</li>
        <li><kbd>Ctrl/Cmd+Z</kbd> undoes the last stroke, <kbd>Ctrl/Cmd+Shift+Z</kbd>
        (or <kbd>Ctrl/Cmd+Y</kbd>) redoes it. Retouch history is per-session, so
        you can undo/redo freely while a session is open.</li>
      </ul>
      <p>All painting works in full-image coordinates through the same
      deep-zoom viewer as everywhere else, so you can zoom in tight for
      precise work.</p>`
  },
  {
    id: 'retouch-flatten',
    group: 'Retouching',
    title: 'Flatten: finishing a retouch session',
    keywords: ['non-destructive', 'new result'],
    html: `
      <p>Retouch strokes accumulate in the session but don't change the
      original result. When you're happy, type a name and click
      <strong>Flatten</strong> — this bakes every stroke into a brand-new
      result, added to the render drawer alongside the original. The result
      you started retouching is left exactly as it was, so a retouch session
      is always undoable at the project level by simply not using its
      flattened output.</p>
      <p>You can retouch the same result again later, or open a fresh session
      on the flattened result itself if you want to layer further manual
      edits on top.</p>`
  },

  // ───────────────────────── Exporting ─────────────────────────
  {
    id: 'export-formats',
    group: 'Exporting',
    title: 'Export formats & bit depth',
    keywords: ['tiff', 'jpeg', 'png', 'compression', 'quality', 'icc', 'exif'],
    html: `
      <p>Click <strong>Export</strong> in the toolbar with a result selected to
      open the export panel. Available formats depend on the result:</p>
      <table>
        <thead><tr><th>Format</th><th>Depth</th><th>Notes</th></tr></thead>
        <tbody>
          <tr><td>TIFF</td><td>8 or 16-bit</td><td>Choose compression: None,
          LZW, or ZIP (lossless either way — this only affects file size and
          write/read speed).</td></tr>
          <tr><td>PNG</td><td>8 or 16-bit</td><td>True 16-bit RGB PNG, not the
          8-bit-with-extra-steps some tools produce.</td></tr>
          <tr><td>JPEG</td><td>8-bit only</td><td>Quality slider 1–100.</td></tr>
          <tr><td>Linear DNG</td><td>16-bit</td><td>Only enabled for a no-bake
          RAW result — see <a href="#linear-dng">Why the output is Linear
          DNG</a>.</td></tr>
        </tbody>
      </table>
      <p>Every export carries the appropriate ICC colour profile and safe
      EXIF/XMP metadata. Choose an <strong>output folder</strong> with the
      folder browser (or type/paste a full path directly), then click
      <strong>Export</strong> in the panel. A progress indicator and, on
      completion, a link to the written file appear in the render drawer
      header.</p>`
  },
  {
    id: 'export-templates',
    group: 'Exporting',
    title: 'Filename templates',
    keywords: ['naming', 'tokens', 'stack_name', 'method', 'seq'],
    html: `
      <p>The <strong>Name template</strong> field builds your output filename
      from tokens, with a live preview shown right below it. The default is
      <code>{stack_name}_{method}</code>.</p>
      <p>Available tokens — see the full list in
      <a href="#filename-tokens">Reference → Filename tokens</a>. Your last-used
      template is remembered per project.</p>
      <p>Characters that aren't valid in a filename (<code>\\ / : * ? " &lt; &gt; |</code>)
      are automatically replaced with an underscore, so a token value can never
      accidentally produce a broken path.</p>`
  },
  {
    id: 'export-companions',
    group: 'Exporting',
    title: 'Float TIFF & depth map companions',
    keywords: ['32-bit float', 'unclamped', 'depth map export'],
    html: `
      <p>Two optional checkboxes in the export panel add an extra file
      alongside your main export, named from the same destination with a
      suffix:</p>
      <ul>
        <li><strong>32-bit float TIFF companion</strong> — an unclamped
        float32 TIFF holding the exact working numbers behind the result,
        including any values below 0 or above the normal 1.0 white point that
        a normal 8/16-bit file would have to clip. This is <em>not</em> a RAW
        file (it doesn't carry camera calibration/mosaic data), but it is the
        most complete non-RAW record of what PlaneFuse actually computed —
        useful for advanced compositing or verifying exposure decisions later.</li>
        <li><strong>16-bit depth map companion</strong> — only available for a
        <a href="#method-dmap">DMap</a> result, this exports the per-pixel
        "which depth was sharpest here" map DMap computed, useful for 3D
        reconstruction, focus-stacking-aware retouching in another tool, or
        just visualising the shape of your subject.</li>
      </ul>`
  },

  // ───────────────────────── RAW & Capture One ─────────────────────────
  {
    id: 'raw-no-bake',
    group: 'RAW & Capture One',
    title: 'What "no-bake" RAW mode means',
    keywords: ['ahd', 'demosaic', 'white balance', 'unit gamma', 'no denoise'],
    html: `
      <p>When you stack RAW files, PlaneFuse decodes them with a single fixed,
      minimal recipe and never applies any of the aesthetic choices a normal
      RAW converter would:</p>
      <ul>
        <li><strong>No white balance</strong> is applied — unit (neutral)
        white balance only.</li>
        <li><strong>No gamma curve</strong> — the data stays in linear light.</li>
        <li><strong>No auto brightness or auto scale.</strong></li>
        <li><strong>No denoise or median filtering.</strong></li>
        <li><strong>No highlight reconstruction.</strong></li>
        <li><strong>No output colour-space conversion.</strong></li>
      </ul>
      <p>The one unavoidable, irreversible step is <strong>AHD demosaic</strong> —
      turning the sensor's raw colour-filter mosaic into full-colour RGB
      pixels — because alignment and stacking both need to combine full RGB
      samples from different focal planes, which a still-mosaiced file can't
      support. Alignment and per-pixel sharpness measurements use their own
      internal proxy values for <em>estimating</em> focus and transforms, but
      fusion always reads and writes the original scene-linear camera RGB
      numbers — no photographic "look" is ever baked into the stacked pixels.</p>
      <p>The practical upshot: you get to make every white balance, exposure,
      tone, colour, noise-reduction, and sharpening decision yourself, in your
      normal raw developer, on the <em>merged, in-focus</em> image — instead
      of inheriting whatever PlaneFuse might have guessed.</p>`
  },
  {
    id: 'linear-dng',
    group: 'RAW & Capture One',
    title: 'Why the output is Linear DNG',
    keywords: ['linearraw', 'dng export', 'mosaic', 'lossless'],
    html: `
      <p>A stacked image cannot be honestly written back out as the original
      Bayer (or other) sensor mosaic — its pixels are combined from several
      different captures and geometric transforms, so there is no real single
      mosaic pattern it could claim to be. Instead, PlaneFuse writes a
      demosaiced, three-channel <strong>LinearRaw DNG</strong> — the DNG
      representation that retains the most real stacked information without
      inventing sensor samples that were never captured.</p>
      <p>The file is 16-bit, losslessly compressed, and keeps camera identity,
      colour matrices, illuminants, as-shot neutral, black/white reference
      levels, orientation, safe EXIF/XMP, and PlaneFuse's own provenance
      record (which frames, which method, which parameters produced it).
      Working values below 0 or above the normal 1.0 white point — which can
      happen during stacking — are encoded with a reversible mapping recorded
      in the file's own metadata, so nothing is silently clipped away.</p>
      <p>Before the file is written to its final destination, PlaneFuse
      reopens it with two independent readers and checks shape, required tags,
      compression, and pixel values match to within one 16-bit code — so a
      corrupted or non-conformant export is caught before you ever see it,
      rather than discovered later inside Capture One.</p>
      <div class="callout note">
        Per-frame focus distance metadata is deliberately dropped from the
        export, because a stack merges many focus distances and there is no
        single correct value to claim.
      </div>`
  },
  {
    id: 'capture-one-checklist',
    group: 'RAW & Capture One',
    title: 'Importing into Capture One: checklist',
    keywords: ['import', 'catalog', 'session', 'workflow'],
    html: `
      <p>PlaneFuse's Linear DNG is written for Capture One Pro (current
      release target: 16.7.5). Suggested workflow:</p>
      <ol>
        <li>Export <code>result.dng</code>; optionally also enable the
        <code>-scene-linear-float.tif</code> companion for advanced work.</li>
        <li>Confirm the PlaneFuse job shows <strong>LibRaw validated</strong>
        before trusting the export.</li>
        <li>Import the DNG into a Capture One Pro Catalog or Session, same as
        any other RAW file.</li>
        <li>Confirm: dimensions and orientation look right; Capture One picked
        either your camera's profile or a generic DNG profile (Capture One
        uses a generic profile when it doesn't natively recognise the camera
        model); white balance is freely editable, not locked; highlight and
        shadow behaviour looks like a normal RAW, not a pre-graded file; there
        is no unexpected style already applied.</li>
        <li>Apply white balance, exposure, curve, colour, noise reduction, and
        sharpening exactly as you would on any other RAW capture. Export a
        TIFF and visually compare it against the PlaneFuse viewer to confirm
        nothing unexpected happened in the round trip.</li>
      </ol>
      <p>Capture One's own DNG support and the current release notes are
      documented officially by Capture One — worth a skim if you hit an
      import quirk that looks specific to their DNG handling rather than to
      the file PlaneFuse produced.</p>`
  },
  {
    id: 'raw-limitations',
    group: 'RAW & Capture One',
    title: 'RAW export limitations, honestly stated',
    keywords: ['caveats', 'not a container', 'focus distance'],
    html: `
      <ul>
        <li>This is a <strong>Linear DNG</strong>, not the original sensor
        mosaic, and not a container that holds your original RAW files inside
        it.</li>
        <li><strong>Per-capture focus distance</strong> is removed from the
        metadata, since a stacked image has no single correct merged value for
        it.</li>
        <li>Capture One does not promise to honour another application's
        embedded adjustments — PlaneFuse embeds provenance and calibration
        metadata, deliberately not a baked photographic look, so there is
        nothing for Capture One to "honour" beyond correct colour and
        exposure math.</li>
        <li>The <a href="#same-camera-rule">same-camera rule</a> intentionally
        rejects stacks whose calibration can't be represented by one honest
        set of DNG metadata — this is a correctness choice, not a missing
        feature.</li>
      </ul>`
  },

  // ───────────────────────── Projects & file safety ─────────────────────────
  {
    id: 'how-projects-work',
    group: 'Projects & file safety',
    title: 'How projects work',
    keywords: ['project.json', 'save', 'scratch project'],
    html: `
      <p>PlaneFuse opens with an untitled <strong>scratch project</strong> —
      you can start adding frames and stacking immediately without any setup.
      Give it a name and click <strong>Save</strong> in the toolbar whenever
      you want it to persist as a named project you can return to later.</p>
      <p>A project remembers its input frames (by reference — see
      <a href="#project-safety">Project safety</a>), every stacked/retouched
      result, job history, and workspace preferences like your last-used
      export template and tonemap setting, in a <code>project.json</code>
      file kept alongside a <code>cache/</code> folder for generated tiles and
      thumbnails.</p>`
  },
  {
    id: 'project-safety',
    group: 'Projects & file safety',
    title: 'Your original photos are never touched',
    keywords: ['delete', 'remove project', 'cache', 'rebuild'],
    html: `
      <p>PlaneFuse reads your source photos from wherever they already live
      on disk and never copies, moves, renames, or deletes them as part of
      normal use. This holds even when you remove a project from PlaneFuse —
      removing a project only <strong>unregisters</strong> it from the app; it
      never deletes your source files.</p>
      <p>The project's <code>cache/</code> folder (tiles, thumbnails) is fully
      rebuildable — safe to clear if you need the disk space, as long as
      PlaneFuse is closed first. Keep <code>project.json</code> together with
      your source files if you want to reopen the exact same project with full
      history later; it's the only part of a project that isn't trivially
      regenerated.</p>
      <div class="callout tip">
        Because sources are read in place, moving or renaming an original
        photo outside PlaneFuse (in Finder/Explorer, another app, etc.) will
        make PlaneFuse unable to find it next time — re-add it from its new
        location, or move it back.
      </div>`
  },

  // ───────────────────────── Troubleshooting ─────────────────────────
  {
    id: 'run-disabled',
    group: 'Troubleshooting',
    title: '"Run" is greyed out',
    keywords: ['cant run', 'disabled button'],
    html: `
      <p>Run stays disabled until every condition below is met — hover it to
      see which one is currently blocking:</p>
      <ul>
        <li>At least one frame is added, and the
        <a href="#validation-panel">validation panel</a> reads <em>Ready to
        stack</em>. Fix or remove any file it names as a blocker.</li>
        <li>At least one stacking method is selected in the toolbar.</li>
        <li>If <strong>Smart frame selection</strong> is enabled, its
        proposal has been reviewed and <strong>Accept proposal</strong>
        clicked — see <a href="#smart-selection">Smart frame selection</a>.</li>
      </ul>`
  },
  {
    id: 'out-of-memory',
    group: 'Troubleshooting',
    title: 'Out of memory / a run is very slow',
    keywords: ['oom', 'crash', 'slow'],
    html: `
      <p>PlaneFuse automatically retries an out-of-memory run with tiled
      processing, then falls back further to tiled CPU processing if needed —
      see <a href="#device-performance">Compute device, GPU, and out-of-memory
      handling</a>. If a run is still uncomfortably slow:</p>
      <ul>
        <li>Close other GPU-heavy applications competing for memory.</li>
        <li>Lower the alignment proxy size (<strong>max long edge</strong>) —
        alignment doesn't need full-resolution pixels to estimate an accurate
        transform.</li>
        <li>For very deep stacks, consider the <a href="#method-slab">Slab</a>
        method, which processes frames in smaller overlapping groups.</li>
      </ul>`
  },
  {
    id: 'dng-export-fails',
    group: 'Troubleshooting',
    title: 'Linear DNG export fails',
    keywords: ['dng error', 'capture one export'],
    html: `
      <p>PlaneFuse never publishes a destination file that fails its own
      checks — if metadata, disk space, TIFF conformance, or the final LibRaw
      pixel-validation step fails, nothing is written rather than writing a
      broken file. Check:</p>
      <ul>
        <li>Every RAW frame in the stack still validates (see the
        <a href="#validation-panel">validation panel</a>) — DNG export is only
        offered for a no-bake RAW result to begin with.</li>
        <li>The destination folder has roughly <strong>twice</strong> the
        uncompressed output size free, since validation briefly needs both the
        candidate file and room to verify it.</li>
        <li>No other process (another app, a sync tool, an open Explorer/Finder
        preview) has the destination path locked.</li>
      </ul>`
  },
  {
    id: 'viewer-issues',
    group: 'Troubleshooting',
    title: 'Histogram or viewer tiles won\'t load',
    keywords: ['blank viewer', 'missing thumbnail', 'moved files'],
    html: `
      <p>Re-select the result — registering a view with the viewer is
      idempotent, so simply clicking it again is often enough to recover.</p>
      <p>If a source or result file was moved outside PlaneFuse after it was
      registered, restore it to its original location (or re-add/rescan it
      from the new one). It's safe to clear just <code>cache/tiles</code>
      while PlaneFuse is closed — tiles rebuild automatically the next time
      you view that image.</p>`
  },
  {
    id: 'port-busy',
    group: 'Troubleshooting',
    title: 'The app won\'t open / port 8425 is busy',
    keywords: ['localhost', 'address in use'],
    html: `
      <p>PlaneFuse serves its interface only on <code>127.0.0.1</code>
      (localhost) — never on your network — for safety. If port 8425 is
      already in use, the native launcher automatically probes the next few
      ports and opens your browser at whichever one it actually bound to, so
      this usually resolves itself without any action.</p>
      <p>If you're running the Docker container, its port mapping is fixed at
      <code>127.0.0.1:8425</code>; stop whatever else is bound to that port
      (or deliberately change both sides of the port mapping in your Docker
      configuration) rather than widening the container's binding to your
      whole network.</p>`
  },

  // ───────────────────────── Reference ─────────────────────────
  {
    id: 'keyboard-shortcuts',
    group: 'Reference',
    title: 'All keyboard shortcuts',
    keywords: ['hotkey list', 'cheat sheet'],
    html: `
      <h3>Viewer</h3>
      <table>
        <tbody>
          <tr><td><kbd>F</kbd></td><td>Fit image to window</td></tr>
          <tr><td><kbd>Z</kbd></td><td>Zoom to 100%</td></tr>
          <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>Step to previous / next input frame</td></tr>
          <tr><td><kbd>\\</kbd> (hold)</td><td>Show comparison image (before/after mode)</td></tr>
          <tr><td><kbd>Esc</kbd></td><td>Exit compare mode</td></tr>
          <tr><td><kbd>?</kbd></td><td>Toggle shortcut cheat-sheet</td></tr>
        </tbody>
      </table>
      <h3>Retouch</h3>
      <table>
        <tbody>
          <tr><td><kbd>Ctrl/Cmd+Z</kbd></td><td>Undo last stroke</td></tr>
          <tr><td><kbd>Ctrl/Cmd+Shift+Z</kbd> or <kbd>Ctrl/Cmd+Y</kbd></td><td>Redo</td></tr>
          <tr><td><kbd>[</kbd> / <kbd>]</kbd></td><td>Shrink / grow brush radius</td></tr>
          <tr><td><kbd>Space</kbd> (hold)</td><td>Pan instead of paint</td></tr>
        </tbody>
      </table>
      <p>Shortcuts are automatically suppressed while a text field, dropdown,
      or other editable control has focus.</p>`
  },
  {
    id: 'glossary',
    group: 'Reference',
    title: 'Glossary',
    keywords: ['definitions', 'terms', 'jargon'],
    html: `
      <table>
        <tbody>
          <tr><td><strong>Focus stacking</strong></td><td>Combining multiple
          photos taken at different focus distances into one image with a
          deeper sharp range than any single frame.</td></tr>
          <tr><td><strong>Focus breathing</strong></td><td>The subtle
          shift/rescale a lens's image undergoes as its focus is racked —
          part of why alignment is needed even on a tripod.</td></tr>
          <tr><td><strong>Alignment</strong></td><td>Estimating and correcting
          the geometric drift between frames before stacking, so the same
          subject point lands on the same pixel in every frame.</td></tr>
          <tr><td><strong>Demosaic</strong></td><td>Converting a RAW sensor's
          mosaic of red/green/blue-filtered pixels into full-colour RGB pixels.
          PlaneFuse uses the AHD demosaic algorithm.</td></tr>
          <tr><td><strong>CFA (colour filter array)</strong></td><td>The
          pattern of colour filters over a sensor's pixels (commonly a Bayer
          pattern) that demosaicing reconstructs full colour from.</td></tr>
          <tr><td><strong>Scene-linear</strong></td><td>Pixel values
          proportional to the actual light the sensor captured, before any
          gamma curve or tone shaping is applied — what a RAW stack is fused
          in and exported as.</td></tr>
          <tr><td><strong>Bit depth</strong></td><td>How many distinct tonal
          levels each colour channel can hold — 8-bit gives 256 levels per
          channel, 16-bit gives 65,536, meaning much smoother gradients and
          more room for edits before banding appears.</td></tr>
          <tr><td><strong>Domain (rendered vs. RAW)</strong></td><td>Which of
          PlaneFuse's two pixel-value pipelines a stack uses — see
          <a href="#rendered-vs-raw">Rendered images vs. camera RAW</a>.</td></tr>
          <tr><td><strong>LinearRaw DNG</strong></td><td>A DNG variant storing
          already-demosaiced, three-channel linear RGB data rather than a
          sensor mosaic — the format PlaneFuse's RAW export writes.</td></tr>
          <tr><td><strong>Provenance</strong></td><td>Metadata a result or
          export carries recording exactly which frames, method, and
          parameters produced it.</td></tr>
          <tr><td><strong>Deep zoom</strong></td><td>Loading and displaying an
          image as a pyramid of resolution levels split into tiles, so panning
          and zooming a very large image stays fast.</td></tr>
          <tr><td><strong>ECC / correlation score</strong></td><td>A numeric
          measure (0–1) of how well two frames' alignment matched — used by
          the quality-exclusion threshold and shown in the alignment
          review.</td></tr>
          <tr><td><strong>Slab</strong></td><td>A sub-stack of consecutive
          frames processed together, used by the Slab method for very deep
          sequences — see <a href="#method-slab">Slab</a>.</td></tr>
        </tbody>
      </table>`
  },
  {
    id: 'filename-tokens',
    group: 'Reference',
    title: 'Export filename tokens',
    keywords: ['template tokens', 'stack_name', 'method', 'frames', 'date', 'seq'],
    html: `
      <table>
        <thead><tr><th>Token</th><th>Expands to</th></tr></thead>
        <tbody>
          <tr><td><code>{stack_name}</code></td><td>The project's name.</td></tr>
          <tr><td><code>{method}</code></td><td>The stacking method that
          produced the selected result (e.g. <code>pmax</code>).</td></tr>
          <tr><td><code>{frames}</code></td><td>Number of frames the result was
          stacked from.</td></tr>
          <tr><td><code>{date}</code></td><td>Today's date, <code>YYYY-MM-DD</code>,
          in your local timezone.</td></tr>
          <tr><td><code>{seq}</code></td><td>A 3-digit sequence number based on
          the result's position among your current project's results (e.g.
          <code>001</code>).</td></tr>
        </tbody>
      </table>
      <p>An unrecognised token (a typo, or a brace that isn't meant as a
      token) is left exactly as written rather than silently dropped, so a
      mistake in the template is easy to spot in the filename preview.</p>`
  }
];
