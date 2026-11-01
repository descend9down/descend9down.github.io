/**
 * Shared WebGL kaleidoscope effect for the DESCEND9DOWN interactive pieces.
 *
 * This is the second generation of this module. The first version (three
 * layers — a counter-rotating pair overlay-blended together, then a slower
 * background layer multiplied in) is still the basis of the look here, but
 * three real problems in it got fixed along the way, in a sandbox ("the
 * Kaleidoscope Lab") before landing back in this file:
 *
 * 1. Rotation was computed as `time * speedMul`, i.e. the angle was
 *    accumulated time multiplied by the CURRENT speed. That's only correct
 *    if speed never changes. The moment it does (a live spin-up ramp, a
 *    slider drag), multiplying a lot of already-elapsed time by a freshly
 *    changed rate produces a phantom burst of angular velocity — a visible
 *    jolt right when the speed change happens. Fixed by integrating
 *    properly: `rotPhase` advances every frame by `dt * speedMul`, so the
 *    instantaneous rotation rate always equals the current speed with no
 *    coupling to how long the piece has already been spinning.
 * 2. Sampling outside the source image returned a flat fallback color,
 *    which shows up as an ugly gray/white ring wherever the fold reaches
 *    past the image's edges — unavoidable for a wide/panoramic piece
 *    inside a square canvas. Switched to the texture's own CLAMP_TO_EDGE
 *    wrap mode instead, which smears the nearest edge pixel and blends in
 *    naturally rather than announcing a hard boundary.
 * 3. The canvas fit used "cover" (crop to fill), which clips the sides off
 *    any piece wider than the canvas is tall. Switched to "contain" (fit
 *    the whole image, letterboxed) so nothing is cropped away.
 *
 * On top of those fixes, a `layerMix` control (0–1) blends between a single
 * clean fold of the actual artwork (0 — maximally recognizable as the
 * piece) and the full multi-layer blended look (1 — the original, more
 * abstract glass effect). This file ships it at 0.46: enough of the
 * layered look to still read as a kaleidoscope, without burying the art
 * under it. `speedMul`, `breatheAmp`, `panAmp`, `glowRadius/Strength`,
 * `vignInner/Outer`, `exposure`, `saturation` and `contrast` are now all
 * real instance properties (not hardcoded shader constants) — nothing in
 * this file exposes a UI for them, but they're there to tune without
 * touching the shader source again.
 *
 * Randomization (so no two activations look identical) is a per-activation
 * seed mixed into each layer's rotation, plus a bounded Lissajous drift
 * across the source artwork and a small amount of slice-count jitter —
 * unchanged in spirit from the first version. The source image is still
 * uploaded to the GPU exactly once, at construction time; every later
 * frame only touches uniforms, never texImage2D again.
 */

class KaleidoscopeFX {
  /**
   * @param {HTMLCanvasElement} canvas - target canvas, sized to its container
   * @param {HTMLImageElement} img - source image, already loaded
   * @param {Object} [opts]
   * @param {[number,number,number]} [opts.glowColor] - center glow RGB, 0-255 each
   * @param {number} [opts.slices] - base slice count (default 9)
   */
  constructor(canvas, img, opts = {}) {
    this.canvas = canvas;
    this.img = img;
    this.glowColor = (opts.glowColor || [160, 60, 255]).map(c => c / 255);

    this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    this.slices = opts.slices || 9;
    this.seed = 0;
    this.panSeedX = 0;
    this.panSeedY = 0;
    this.sliceJitterFront = 0;
    this.sliceJitterBack = 0;

    // Tuned defaults from the Kaleidoscope Lab, baked in directly since
    // these pieces have no UI of their own for any of them.
    this.speedMul = 1.45;
    // rotPhase is the actual integrated rotation angle — see the module
    // header's fix #1. Advanced each frame by dt*speedMul in _loop().
    this.rotPhase = 0;
    this.panSpeed = 2.0;
    this.breatheAmp = 1.15;
    this.panAmp = 1;
    this.glowRadius = 26;
    this.glowStrength = 0.6;
    this.vignInner = 0.80;
    this.vignOuter = 0.90;
    this.exposure = 1.16;
    this.saturation = 1.35;
    this.contrast = 1.2;
    // 0 = one clean fold of the actual art, 1 = the full blended
    // multi-layer look. See the module header for why this sits at 0.46.
    this.layerMix = 0.46;

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
      uniform float u_rotPhase;
      uniform float u_seed;
      uniform float u_panSeedX;
      uniform float u_panSeedY;
      uniform float u_panSpeed;
      uniform vec3 u_glowColor;
      uniform float u_breatheAmp;
      uniform float u_panAmp;
      uniform float u_glowRadius;
      uniform float u_glowStrength;
      uniform float u_vignInner;
      uniform float u_vignOuter;
      uniform float u_exposure;
      uniform float u_saturation;
      uniform float u_contrast;
      uniform float u_layerMix;
      #define PI 3.14159265359

      vec3 sampleTex(vec2 px){
        // Outside the image, let the texture's CLAMP_TO_EDGE wrap mode
        // smear the nearest edge pixel instead of substituting a flat
        // fallback color — a flat color reads as an ugly gray/white
        // ring wherever the fold samples past the image bounds
        // (unavoidable for a wide panoramic piece in a square frame);
        // edge-smearing blends in naturally instead.
        vec2 uv = (px - u_imgOrigin) / u_imgSize;
        return texture2D(u_tex, uv).rgb;
      }

      // Bounded Lissajous drift across the source artwork — sin/cos on
      // two frequencies rather than a straight line, so it endlessly
      // wanders without ever needing to wrap or jump.
      vec2 ringPan(float ampFrac, float freqX, float freqY, float phase){
        float t = u_time * u_panSpeed;
        float px = sin(t * freqX + u_panSeedX + phase);
        float py = cos(t * freqY + u_panSeedY + phase * 1.3);
        return vec2(px, py) * u_imgSize * ampFrac * u_panAmp;
      }

      // Independently rotated whole copies of the artwork, folded into
      // wedges — deliberate seam mismatches between wedges are what give
      // this its glass-like feel rather than reading as one clean,
      // static mandala.
      vec3 foldSample(vec2 p, vec2 center, float slices, float time, float scale, vec2 pan){
        vec2 ps = p / scale;
        float theta = atan(ps.y, ps.x);
        float slice = 2.0 * PI / slices;
        float k = floor((theta - time) / slice + 0.5);
        float wedgeAngle = k * slice + time;
        float ca = cos(-wedgeAngle), sa = sin(-wedgeAngle);
        vec2 rotated = vec2(ca * ps.x - sa * ps.y, sa * ps.x + ca * ps.y);
        return sampleTex(center + rotated + pan);
      }

      vec3 overlayBlend(vec3 a, vec3 b){
        vec3 lo = 2.0 * a * b;
        vec3 hi = 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
        return mix(lo, hi, step(0.5, a));
      }

      vec3 multiplyBlend(vec3 a, vec3 b){
        return a * b;
      }

      vec3 liftShadows(vec3 c){
        return pow(clamp(c, 0.0, 1.0), vec3(0.56));
      }

      void main(){
        vec2 fragPx = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
        vec2 center = u_resolution * 0.5;
        vec2 p = fragPx - center;
        float r = length(p);
        float fullR = min(u_resolution.x, u_resolution.y) * 0.5;

        float breathe1 = 1.0  * (1.0 + 0.10 * u_breatheAmp * sin(u_time * 0.07 + u_seed * 0.4));
        float breathe2 = 0.95 * (1.0 + 0.08 * u_breatheAmp * sin(u_time * 0.05 + u_seed * 0.6 + 2.0));

        vec2 pan1 = ringPan(0.30, 0.050, 0.037, u_seed * 0.9);
        vec2 pan2 = ringPan(0.30, 0.041, 0.033, u_seed * 1.7 + 2.0);
        vec2 pan3 = ringPan(0.14, 0.026, 0.019, u_seed * 2.6 + 5.0);

        vec3 layer1 = liftShadows(foldSample(p, center, u_slices0,  u_rotPhase + u_seed * 2.0, breathe1, pan1));
        vec3 layer2 = liftShadows(foldSample(p, center, u_slices0, -u_rotPhase + u_seed * 5.0, breathe2, pan2));
        vec3 layer3 = liftShadows(foldSample(p, center, u_slices1,  u_rotPhase * 0.4 + u_seed * 7.0, 0.5, pan3));

        vec3 complexCol = 0.5 * (overlayBlend(layer1, layer2) + overlayBlend(layer2, layer1));

        float layer3Lum = dot(layer3, vec3(0.299, 0.587, 0.114));
        vec3 layer3ForBlend = mix(layer3, vec3(layer3Lum), 0.85);
        complexCol = multiplyBlend(complexCol, layer3ForBlend);

        // A single, un-blended fold of the actual source pixels — no
        // second rotated copy overlaid on top, no third layer multiplied
        // in on top of that. This is what keeps the piece recognizable
        // as itself: fewer compounding blend passes means more of the
        // original color and shape survives the fold. u_layerMix dials
        // between this and the fuller, more abstract layer stack above.
        vec3 simpleCol = pow(clamp(foldSample(p, center, u_slices0, u_rotPhase + u_seed * 2.0, breathe1, pan1), 0.0, 1.0), vec3(0.85));

        vec3 col = mix(simpleCol, complexCol, u_layerMix);

        float glow = exp(-r / (u_glowRadius * (u_resolution.x / 480.0)));
        col = mix(col, u_glowColor, glow * u_glowStrength);

        float vign = smoothstep(fullR * u_vignInner, fullR * u_vignOuter, r);
        col = mix(col, vec3(0.0), vign);

        col *= u_exposure;

        vec3 gray = vec3(dot(col, vec3(0.299, 0.587, 0.114)));
        col = mix(gray, col, u_saturation);
        col = clamp((col - 0.5) * u_contrast + 0.5, 0.0, 1.0);

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

    const names = ['resolution', 'tex', 'imgSize', 'imgOrigin', 'time', 'rotPhase', 'seed',
      'panSeedX', 'panSeedY', 'panSpeed', 'glowColor', 'breatheAmp', 'panAmp', 'glowRadius',
      'glowStrength', 'vignInner', 'vignOuter', 'exposure', 'saturation', 'contrast', 'layerMix'];
    names.forEach(n => { this._u[n] = gl.getUniformLocation(program, 'u_' + n); });
    this._u.slices = [0, 1].map(i => gl.getUniformLocation(program, `u_slices${i}`));

    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // The one and only texture upload this module ever does. Rounded to
    // the nearest power of two on each axis so gl.generateMipmap works
    // (WebGL1 refuses to mipmap NPOT textures) — without mipmaps the
    // outer, zoomed-out rings alias badly against this artwork's dense,
    // high-frequency texture.
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

    // "Contain" fit (min, not max) so wide panoramic pieces show their
    // full width instead of having the sides cropped off to fill a
    // square frame. Any resulting slack shows as the edge-clamped smear
    // from sampleTex() rather than losing real artwork off the edges.
    const iW = this.img.naturalWidth, iH = this.img.naturalHeight;
    const scale = Math.min(w / iW, h / iH);
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

  randomize() {
    this.seed = Math.random() * 1000;
    this.panSeedX = Math.random() * 1000;
    this.panSeedY = Math.random() * 1000;
    this.sliceJitterFront = Math.round((Math.random() - 0.5) * 4);
    this.sliceJitterBack = Math.round((Math.random() - 0.5) * 4);
    if (!this.active) this._draw();
    return this.seed;
  }

  start() {
    if (this.active || !this.ready) return;
    // Fresh rotational phase, drift starting point and slice jitter per
    // activation — no texture rebuild, just a handful of new uniform
    // values, so restarting genuinely samples a different journey across
    // the artwork rather than replaying the same one from angle zero.
    this.randomize();
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
    const dtScaled = dt * 0.24;
    this.time += dtScaled;
    // Integrate rotation properly: advance by the CURRENT speed each
    // frame, rather than multiplying accumulated time by speed after the
    // fact. See the module header's fix #1 for why that distinction
    // matters.
    this.rotPhase += dtScaled * this.speedMul;
    this._draw();
    this.raf = requestAnimationFrame((t) => this._loop(t));
  }

  /** Per-frame draw. Deliberately does no texture upload — only uniform
   *  writes (cheap) and rebinding the one texture already resident on
   *  the GPU since construction. */
  _draw() {
    const gl = this.gl;
    if (!this.ready) return;
    gl.useProgram(this.program);
    gl.uniform2f(this._u.resolution, this.canvas.width, this.canvas.height);
    gl.uniform2f(this._u.imgSize, this.imgSize[0], this.imgSize[1]);
    gl.uniform2f(this._u.imgOrigin, this.imgOrigin[0], this.imgOrigin[1]);
    gl.uniform1f(this._u.slices[0], Math.max(4, this.slices + this.sliceJitterFront));
    gl.uniform1f(this._u.slices[1], Math.max(3, Math.round(this.slices * 0.5) + this.sliceJitterBack));
    gl.uniform1f(this._u.time, this.time);
    gl.uniform1f(this._u.rotPhase, this.rotPhase);
    gl.uniform1f(this._u.seed, this.seed);
    gl.uniform1f(this._u.panSeedX, this.panSeedX);
    gl.uniform1f(this._u.panSeedY, this.panSeedY);
    gl.uniform1f(this._u.panSpeed, this.panSpeed);
    gl.uniform3f(this._u.glowColor, this.glowColor[0], this.glowColor[1], this.glowColor[2]);
    gl.uniform1f(this._u.breatheAmp, this.breatheAmp);
    gl.uniform1f(this._u.panAmp, this.panAmp);
    gl.uniform1f(this._u.glowRadius, this.glowRadius);
    gl.uniform1f(this._u.glowStrength, this.glowStrength);
    gl.uniform1f(this._u.vignInner, this.vignInner);
    gl.uniform1f(this._u.vignOuter, this.vignOuter);
    gl.uniform1f(this._u.exposure, this.exposure);
    gl.uniform1f(this._u.saturation, this.saturation);
    gl.uniform1f(this._u.contrast, this.contrast);
    gl.uniform1f(this._u.layerMix, this.layerMix);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this._u.tex, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
