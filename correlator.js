// correlator.js - AudioWorklet for echo cancellation via reference correlation
class Correlator extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = 480; // 10ms @48k (or proportionally less at 16k)
  }

  _rms(buf) {
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    return Math.sqrt(s / buf.length);
  }

  _corr(x, y) {
    // Cosine similarity = normalized correlation
    let xy = 0, xx = 0, yy = 0;
    for (let i = 0; i < x.length; i++) {
      const a = x[i], b = y[i];
      xy += a * b;
      xx += a * a;
      yy += b * b;
    }
    return xy / (Math.sqrt(xx * yy) + 1e-9);
  }

  process(inputs, outputs) {
    const mic = inputs[0][0];       // Microphone input
    const ref = inputs[1]?.[0];     // Reference (TTS output)
    const out = outputs[0][0];      // Pass-through output

    if (!mic) return true;

    // Pass-through mic
    if (out) {
      for (let i = 0; i < mic.length; i++) out[i] = mic[i];
    }

    // Calculate correlation and RMS levels
    if (ref && ref.length === mic.length) {
      const corr = this._corr(mic, ref);   // -1..1
      const micRms = this._rms(mic);
      const refRms = this._rms(ref);
      this.port.postMessage({ 
        corr: Math.max(0, corr), 
        micRms, 
        refRms 
      });
    } else {
      const micRms = this._rms(mic);
      this.port.postMessage({ corr: 0, micRms, refRms: 0 });
    }
    return true;
  }
}

registerProcessor('correlator', Correlator);