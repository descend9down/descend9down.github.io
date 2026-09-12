/**
 * Shared WebGL kaleidoscope effect for the DESCEND9DOWN interactive pieces.
 *
 * Earlier version of this module enforced a mathematically seamless
 * mirror-fold (each wedge a perfect reflection of its neighbor). That
 * turned out to be the wrong goal: the mismatched seams in the original
 * per-piece Canvas 2D implementation — where each wedge independently
 * showed a rotated whole copy of the image, not a true reflection — were
 * what gave it a trippy, glass-like feeling. A perfectly seamless fold
 * just reads as one clean, static mandala.
 *
 * A later version reconstructed that "independently rotated copies" look
 * analytically and composited three layers — first screen-blended (read as
 * washed out and moired in motion), then as opaque hard-edged radial zones
 * sampling a pre-fragmented, dark-stroked copy of the artwork (read as
 * black-grouted stained glass), then as four nested radial rings each with
 * its own zoom depth (closer to the target reference, but radius-based
 * zone boundaries — even softened ones — read as visible seams and cusps
 * between rings, and any gap between a ring's valid sample area and its
 * neighbor's shows as a black gap; not the genuine full-frame overlap of
 * real glass).
 *
 * This version drops radial zones entirely. Three layers, each covering
 * the FULL frame (no radius cutoff of any kind), each independently
 * mirror-folded with its own slice count, rotation direction/speed and
 * zoom: layer 1 and layer 2 share a slice count and zoom but spin in
 * opposite directions (the counter-rotating pair that creates the
 * "turning against itself" complexity); layer 3 has fewer, larger
 * slices, is zoomed out further and rotates slower, reading as a
 * background sitting behind the other two. They're combined with real
 * per-pixel blending — overlay for the counter-rotating pair, then
 * multiply to bring in the background layer (multiply is what makes it
 * recede/darken through the front pair rather than compete with it) —
 * never screen blending, which is what caused an earlier attempt's
 * washed-out white-blob look. Every pixel samples all three layers, so
 * there are no seams, no cusps and no gaps: it's genuine full-frame
 * merging, not adjacent zones.
 *
 * Randomization (so no two activations look identical) is a single
 * random phase seed mixed into each layer's rotation, not a regenerated
 * texture — which also means the source image is uploaded to the GPU
 * exactly once, ever, at construction time. See _draw()'s comment for
 * why that one-time upload matters.
 *
 * Each layer's sample position also slowly pans across the source
 * artwork (a bounded Lissajous drift, not a straight line — it never
 * needs to wrap or jump) and the front pair gently breathes in and out
 * in zoom, so the actual shapes being folded keep evolving instead of
 * just spinning in place. The pan's starting position and speed are
 * randomized per activation (not just the rotational phase), so
 * restarting the effect samples a genuinely different journey across
 * the artwork, not just a different angle on the same fixed crop.
 */

class KaleidoscopeFX {
  /**
   * @param {HTMLCanvasElement} canvas - target canvas, sized to its container
   * @param {HTMLImageElement} img - source image, already loaded
   * @param {Object} [opts]
   * @param {[number,number,number]} [opts.glowColor] - center glow RGB, 0-255 each
   */
  constructor(canvas, img, opts = {}) {
    this.canvas = canvas;
    this.img = img;
    this.glowColor = (opts.glowColor || [160, 60, 255]).map(c => c / 255);

    this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    this.slices = 8;
    this.seed = 0;
    this.panSeedX = 0;
    this.panSeedY = 0;
    this.panSpeed = 1;
    this.sliceJitterFront = 0;
    this.sliceJitterBack = 0;
    this.active = false;
    this.time = 0;
    this.lastFrame = 0;
    this.raf = null;
    this.ready = false;

    this._u = {};
    if (this.gl) this._initGL();
  }

  _compileShader(type, src) {
    const gl = this.gl;
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('KaleidoscopeFX shader error:', gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  _initGL() {
    const gl = this.gl;
    const vsSrc = 'attribute vec2 a_pos; void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }';
    const fsSrc = `
      precision highp float;
      uniform vec2 u_resolution;
      uniform sampler2D u_tex;
      uniform vec2 u_imgSize;
      uniform vec2 u_imgOrigin;
      uniform float u_slices0;
      uniform float u_slices1;
      uniform float u_time;
      uniform float u_seed;
      uniform float u_panSeedX;
      uniform float u_panSeedY;
      uniform float u_panSpeed;
      uniform vec3 u_glowColor;
      #define PI 3.14159265359

      // fallback is what to return when the sample falls outside the
      // source image's mapped rectangle. This MUST be the blend's own
      // neutral color, not black: whichever layer runs off its image
      // edge first (the zoomed-out background layer, with the least
      // wedges, hits this soonest — well within the visible circle) was
      // returning black, which a multiply blend turns into "zero out
      // the entire composite here" and an overlay blend also collapses
      // to black for. That imposed that one layer's own image-edge
      // shape — a rectangle sampled through a low wedge count reads as
      // a rotated square — onto the whole frame, which is what was
      // actually producing the "diamond/rotated square" frame: not the
      // vignette (which is a real, verified circle), but this.
      vec3 sampleTex(vec2 px, vec3 fallback){
        vec2 uv = (px - u_imgOrigin) / u_imgSize;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return fallback;
        return texture2D(u_tex, uv).rgb;
      }

      // Bounded Lissajous drift across the source artwork — sin/cos on
      // two frequencies rather than a straight line, so it endlessly
      // wanders without ever needing to wrap or jump. ampFrac scales
      // with u_imgSize so it stays a sane fraction of the actual image
      // regardless of canvas size. u_panSeedX/Y (randomized per
      // activation, see start()) shift the phase so a fresh activation
      // starts the drift from a different point and, since freqX/freqY
      // are irrational-ish per-ring constants, traces a different path
      // over time too — not just a different angle on the same journey.
      vec2 ringPan(float ampFrac, float freqX, float freqY, float phase){
        float t = u_time * u_panSpeed;
        float px = sin(t * freqX + u_panSeedX + phase);
        float py = cos(t * freqY + u_panSeedY + phase * 1.3);
        return vec2(px, py) * u_imgSize * ampFrac;
      }

      // Same "independently rotated whole copies" reconstruction as
      // before (wedges don't line up with their neighbors — deliberate),
      // but reading straight from the one artwork texture. scale zooms
      // the sample in (>1) or out (<1) so each ring can crop a different
      // depth of the same image; pan shifts which part of the artwork
      // that crop is centered on, so the folded content itself evolves
      // over time instead of just spinning in place.
      //
      // This deliberately does NOT also nudge the sample position per
      // color channel here to fake fringing at wedge seams: the artwork
      // is dense, high-frequency texture with no smooth flat regions, so
      // a sub-pixel shift changes the sampled color almost everywhere,
      // not just at hard edges — it reads as noise over the whole ring,
      // not a fringe at its seams. Fringing instead comes only from
      // ringColor()'s boundary blend below, which mixes two genuinely
      // different, already-correct ring images near their shared edge.
      vec3 foldSample(vec2 p, vec2 center, float slices, float time, float scale, vec2 pan, vec3 fallback){
        vec2 ps = p / scale;
        float theta = atan(ps.y, ps.x);
        float slice = 2.0 * PI / slices;
        float k = floor((theta - time) / slice + 0.5);
        float wedgeAngle = k * slice + time;
        float ca = cos(-wedgeAngle), sa = sin(-wedgeAngle);
        vec2 rotated = vec2(ca * ps.x - sa * ps.y, sa * ps.x + ca * ps.y);
        return sampleTex(center + rotated + pan, fallback);
      }

      // Photoshop-style overlay: darkens where the base is dark,
      // lightens where it's light, preserving contrast from both inputs
      // rather than just averaging or adding brightness. Used for the
      // counter-rotating pair so their overlap reads as real interference,
      // not a flat cross-fade.
      vec3 overlayBlend(vec3 a, vec3 b){
        vec3 lo = 2.0 * a * b;
        vec3 hi = 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
        return mix(lo, hi, step(0.5, a));
      }

      // Multiply: never brightens, only darkens — the standard way
      // stacked colored glass behaves. Used to bring in the background
      // layer so it visually recedes behind the front pair instead of
      // competing with it for brightness (which is what screen blending
      // did — see the module header).
      vec3 multiplyBlend(vec3 a, vec3 b){
        return a * b;
      }

      // This artwork is dark and moody — mostly-black backgrounds with
      // vivid highlights — so multiplying/overlaying several samples of
      // it together, unadjusted, crushes almost the whole frame to black
      // and leaves only sparse bright slivers where highlights from all
      // layers happen to coincide (the opposite failure from screen
      // blending's washed-out look, but just as wrong: not genuine
      // full-frame density). Lifting shadows before combining gives
      // overlay/multiply actual midtone material to work with everywhere,
      // so the merge reads as dense overlapping glass across the whole
      // frame instead of a starburst of rare bright spots.
      vec3 liftShadows(vec3 c){
        return pow(clamp(c, 0.0, 1.0), vec3(0.56));
      }

      void main(){
        vec2 fragPx = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
        vec2 center = u_resolution * 0.5;
        vec2 p = fragPx - center;
        float r = length(p);
        // Half the canvas's shorter side — the true, aspect-safe max
        // radius of a circle that fits inside the frame in every
        // direction, cardinal included. The vignette below is anchored
        // to THIS, not to how far the pattern's own bright content
        // happens to reach.
        float fullR = min(u_resolution.x, u_resolution.y) * 0.5;

        // Gentle in/out zoom breathing on the front counter-rotating
        // pair — small and slow enough to read as a pulse, not a
        // rack-focus. The background layer doesn't breathe: it's already
        // zoomed out and reaches further per screen pixel, so it has the
        // least headroom before pan+rotation would push a sample outside
        // the uploaded texture.
        float breathe1 = 1.0  * (1.0 + 0.10 * sin(u_time * 0.07 + u_seed * 0.4));
        float breathe2 = 0.95 * (1.0 + 0.08 * sin(u_time * 0.05 + u_seed * 0.6 + 2.0));

        // Wider than before: a narrow drift range meant each layer
        // tended to stay parked on whichever crop of the artwork it
        // started near, so if that happened to be a blue-leaning region
        // (as it was for layer1 in the case that exposed the overlay
        // bug above), it stayed blue-leaning for the whole activation.
        // More range means each layer actually traverses more of the
        // piece's real color variety over the course of one activation.
        vec2 pan1 = ringPan(0.30, 0.050, 0.037, u_seed * 0.9);
        vec2 pan2 = ringPan(0.30, 0.041, 0.033, u_seed * 1.7 + 2.0);
        vec2 pan3 = ringPan(0.14, 0.026, 0.019, u_seed * 2.6 + 5.0);

        // Out-of-bounds fallback colors, chosen so that AFTER
        // liftShadows (pow(c, 0.56)) they land exactly on each blend's
        // true identity value: 0.290^0.56 = 0.5 (overlayBlend(0.5,b)=b,
        // its neutral), and 1.0^0.56 = 1.0 (multiplyBlend(1,b)=b, its
        // neutral). So a layer running off the image edge now
        // contributes nothing to the composite there, instead of
        // forcing black through the blend and imposing its own
        // image-edge shape on the whole frame.
        vec3 overlayNeutral = vec3(0.290);
        vec3 multiplyNeutral = vec3(1.0);

        // Layer 1 and layer 2: same slice count and similar zoom,
        // opposite rotation direction — the counter-rotating pair.
        // Layer 3: fewer, larger slices, zoomed out further, rotating
        // slower — a background sitting behind the pair.
        vec3 layer1 = liftShadows(foldSample(p, center, u_slices0,  u_time * 1.0 + u_seed * 2.0, breathe1, pan1, overlayNeutral));
        vec3 layer2 = liftShadows(foldSample(p, center, u_slices0, -u_time * 1.0 + u_seed * 5.0, breathe2, pan2, overlayNeutral));
        vec3 layer3 = liftShadows(foldSample(p, center, u_slices1,  u_time * 0.4 + u_seed * 7.0, 0.5,      pan3, multiplyNeutral));

        // Real per-pixel blending across the FULL frame — every pixel
        // samples all three layers, so there's no radius cutoff, no
        // seam and no gap anywhere, unlike the zone/ring approach this
        // replaced.
        //
        // overlayBlend(a,b) is NOT symmetric: it picks its lo/hi branch
        // per-channel based on a alone, so whichever layer is passed
        // first gets outsized control over which colors can survive —
        // confirmed by a controlled test where layer1 (sampling a
        // blue-heavy crop) and layer2 (sampling a magenta-heavy crop of
        // the SAME image) combined via overlayBlend(layer1, layer2)
        // crushed magenta from layer2's own 44% down to 19%, i.e. this
        // was measurably pulling every piece toward whichever hue
        // layer1's pan happens to land on, not a fair merge of both.
        // Averaging both orderings removes that arbitrary bias.
        vec3 col = 0.5 * (overlayBlend(layer1, layer2) + overlayBlend(layer2, layer1));

        // layer3 is sampled zoomed out 2x through few wedges, which
        // means each screen pixel covers a much wider, heavily
        // mip-blurred swath of the source image than layer1/2 do — it's
        // effectively a low-resolution, spatially-averaged copy. Since
        // multiply can only ever REMOVE color a layer doesn't have, and
        // averaging together many differently-hued pixels usually
        // converges toward whichever hue is most prevalent overall
        // (here, blue/violet) rather than preserving the full spread,
        // multiplying that blurred average in at full saturation was
        // acting as a whole-frame tint toward that one dominant hue —
        // crushing out the piece's own minority colors (e.g. Beholder's
        // magenta) everywhere, not just where layer3 has real texture.
        // Mostly desaturating it before the multiply keeps its
        // contribution to what it's actually there for — shape and
        // depth from a third, slower-moving layer — without also
        // forcing its own averaged hue onto layer1/2's true colors.
        float layer3Lum = dot(layer3, vec3(0.299, 0.587, 0.114));
        vec3 layer3ForBlend = mix(layer3, vec3(layer3Lum), 0.85);
        col = multiplyBlend(col, layer3ForBlend);

        // Bright glow concentrated at dead center, additive so the
        // pattern brightens and radiates from it rather than being
        // replaced by a flat tint. Turned down significantly (from
        // 1.15 intensity and a 42px falloff radius) — at full strength
        // this alone was enough to read as a wash of the glow color
        // (mostly magenta/purple across the pieces that use it) over
        // the whole frame rather than a subtle accent at dead center;
        // each piece's own artwork colors should read clearly with
        // only a small, tight highlight at the very core.
        float glow = exp(-r / (26.0 * (u_resolution.x / 480.0)));
        col += u_glowColor * glow * 0.55;

        // Circular porthole cutoff. This previously started past fullR
        // (>1.0x) and finished even further out — since fullR itself
        // could already exceed the canvas's own half-width/height, the
        // fade frequently never engaged before hitting the frame edge,
        // so the pattern's own cross/star-shaped brightness (not a
        // circle) was what defined the visible silhouette. Anchoring
        // both ends inside fullR (the true max safe radius) fixed the
        // gross shape, but a wide transition band still let the
        // pattern's own bright spots pierce through it unevenly —
        // visible as thin spikes breaking the circular edge in a few
        // directions. Narrowed here so the cutoff is crisp enough that
        // content brightness can't leak past it directionally.
        float vign = smoothstep(fullR * 0.80, fullR * 0.90, r);
        col = mix(col, vec3(0.0), vign);

        // Overall exposure lift — same look, just brighter.
        col *= 1.16;

        // Bold, high-contrast glass rather than a soft filtered photo —
        // a static per-pixel push, not a blend of anything.
        vec3 gray = vec3(dot(col, vec3(0.299, 0.587, 0.114)));
        col = mix(gray, col, 1.35);
        col = clamp((col - 0.5) * 1.2 + 0.5, 0.0, 1.0);

        gl_FragColor = vec4(col, 1.0);
      }
    `;

    const vs = this._compileShader(gl.VERTEX_SHADER, vsSrc);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSrc);
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('KaleidoscopeFX link error:', gl.getProgramInfoLog(program));
      return;
    }
    this.program = program;
    gl.useProgram(program);

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(program, 'a_pos');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    this._u.resolution = gl.getUniformLocation(program, 'u_resolution');
    this._u.tex = gl.getUniformLocation(program, 'u_tex');
    this._u.imgSize = gl.getUniformLocation(program, 'u_imgSize');
    this._u.imgOrigin = gl.getUniformLocation(program, 'u_imgOrigin');
    this._u.slices = [0, 1].map(i => gl.getUniformLocation(program, `u_slices${i}`));
    this._u.time = gl.getUniformLocation(program, 'u_time');
    this._u.seed = gl.getUniformLocation(program, 'u_seed');
    this._u.panSeedX = gl.getUniformLocation(program, 'u_panSeedX');
    this._u.panSeedY = gl.getUniformLocation(program, 'u_panSeedY');
    this._u.panSpeed = gl.getUniformLocation(program, 'u_panSpeed');
    this._u.glowColor = gl.getUniformLocation(program, 'u_glowColor');

    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // The one and only texture upload this module ever does. The source
    // artwork is downscaled once to a GPU- and mobile-safe cap (source
    // files here run up to ~4750px on a side, past what some devices'
    // MAX_TEXTURE_SIZE allows) and pushed to the GPU right here at
    // construction time. Nothing after this point ever calls
    // texImage2D again — activations just change a rotation/seed
    // uniform, not the pixels underneath it.
    //
    // Rounded to the nearest power of two on each axis (independently,
    // so a non-square image just gets stretched slightly, not cropped —
    // harmless since UV coordinates are normalized 0-1 either way) so
    // gl.generateMipmap works: WebGL1 refuses to mipmap NPOT textures.
    // Without mipmaps, the outer rings (sampled at under 1x scale — see
    // foldSample's `scale` param) alias badly against this artwork's
    // dense, high-frequency texture, reading as colored moire noise
    // instead of the zoomed-out pattern it's supposed to be.
    const maxDim = 2048;
    const iW = img.naturalWidth, iH = img.naturalHeight;
    const scale = Math.min(1, maxDim / Math.max(iW, iH));
    const nearestPow2 = (n) => Math.pow(2, Math.round(Math.log2(Math.max(1, n))));
    const uploadCanvas = document.createElement('canvas');
    uploadCanvas.width = nearestPow2(iW * scale);
    uploadCanvas.height = nearestPow2(iH * scale);
    uploadCanvas.getContext('2d').drawImage(img, 0, 0, uploadCanvas.width, uploadCanvas.height);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, uploadCanvas);
    gl.generateMipmap(gl.TEXTURE_2D);

    this.ready = true;
    this.resize();
  }

  /** Call after the canvas element's on-screen size changes. */
  resize() {
    if (!this.gl) return;
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(this.canvas.clientWidth * dpr) || this.canvas.width;
    const h = Math.round(this.canvas.clientHeight * dpr) || this.canvas.height;
    this.canvas.width = w;
    this.canvas.height = h;
    gl.viewport(0, 0, w, h);

    const iW = this.img.naturalWidth, iH = this.img.naturalHeight;
    const scale = Math.max(w / iW, h / iH) * 0.8;
    this.imgSize = [iW * scale, iH * scale];
    this.imgOrigin = [w / 2 - this.imgSize[0] / 2, h / 2 - this.imgSize[1] / 2];
  }

  setSlices(n) {
    this.slices = n;
  }

  /** @returns {boolean} the new active state */
  toggle() {
    this.active ? this.stop() : this.start();
    return this.active;
  }

  start() {
    if (this.active || !this.ready) return;
    // Fresh rotational phase, drift starting point and drift speed per
    // activation — no texture rebuild, just a handful of new uniform
    // values, so restarting genuinely samples a different journey across
    // the artwork rather than replaying the same one from angle zero.
    this.seed = Math.random() * 1000;
    this.panSeedX = Math.random() * 1000;
    this.panSeedY = Math.random() * 1000;
    this.panSpeed = 0.6 + Math.random() * 0.8;
    // Jitter the actual rendered slice counts around the user-chosen
    // base, independently for the counter-rotating pair and the
    // background layer. Without this, both were deterministic functions
    // of `slices` alone (front = slices, back = round(slices*0.5)) —
    // only their rotation varied by seed, so every activation was built
    // from the exact same wedge-count skeleton and always converged on
    // the same macro shape family, just spun to a different angle. Now
    // a given base slice count can render as a visibly different
    // pattern (fewer/more, sharper/softer wedges) from one activation
    // to the next, not just a rotated copy of the same one.
    this.sliceJitterFront = Math.round((Math.random() - 0.5) * 4); // -2..+2
    this.sliceJitterBack = Math.round((Math.random() - 0.5) * 4);  // -2..+2, independent
    this.active = true;
    this.lastFrame = performance.now();
    this.raf = requestAnimationFrame((t) => this._loop(t));
  }

  stop() {
    this.active = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  _loop(now) {
    if (!this.active) return;
    const dt = (now - this.lastFrame) / 1000;
    this.lastFrame = now;
    this.time += dt * 0.24; // matches the original's perceived rotation speed
    this._draw();
    this.raf = requestAnimationFrame((t) => this._loop(t));
  }

  /** Per-frame draw. Deliberately does no texture upload — only uniform
   *  writes (cheap) and rebinding the one texture already resident on
   *  the GPU since construction. Re-uploading full-resolution image data
   *  every frame is what silently killed earlier prototypes (sustained
   *  upload pressure caused WebGL context loss after a few seconds);
   *  keep it that way — texImage2D belongs only in _initGL(), never here
   *  or in start()/_loop(). */
  _draw() {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform2f(this._u.resolution, this.canvas.width, this.canvas.height);
    gl.uniform2f(this._u.imgSize, this.imgSize[0], this.imgSize[1]);
    gl.uniform2f(this._u.imgOrigin, this.imgOrigin[0], this.imgOrigin[1]);
    // Layer 1 & 2 (the counter-rotating pair) share this slice count;
    // layer 3 (background) gets fewer, larger slices. Both are jittered
    // per-activation (see start()) around the user-chosen base.
    gl.uniform1f(this._u.slices[0], Math.max(4, this.slices + this.sliceJitterFront));
    gl.uniform1f(this._u.slices[1], Math.max(3, Math.round(this.slices * 0.5) + this.sliceJitterBack));
    gl.uniform1f(this._u.time, this.time);
    gl.uniform1f(this._u.seed, this.seed);
    gl.uniform1f(this._u.panSeedX, this.panSeedX);
    gl.uniform1f(this._u.panSeedY, this.panSeedY);
    gl.uniform1f(this._u.panSpeed, this.panSpeed);
    gl.uniform3f(this._u.glowColor, this.glowColor[0], this.glowColor[1], this.glowColor[2]);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this._u.tex, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
