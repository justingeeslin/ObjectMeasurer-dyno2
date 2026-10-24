(() => {
  'use strict';
  const form = document.querySelector('#demo-form');
  if (!form) return;
  const $ = (selector) => document.querySelector(selector);
  const isMeasure = form.dataset.kind === 'measure';
  const blobs = [];
  const shellQuote = (value) => "'" + String(value).replaceAll("'", "'\"'\"'") + "'";
  let curl = '';
  let busy = false;

  function options() {
    const params = new URLSearchParams();
    for (const [key, value] of new FormData(form)) {
      if (typeof value === 'string' && value.trim() !== '') params.set(key, value);
    }
    if (isMeasure) {
      const mode = $('#reference-mode').value;
      if (mode === 'letter') params.set('reference_size_mm', '215.9,279.4');
      if (mode === 'compact') params.set('reference_size_mm', `${$('#reference-width').value},${$('#reference-height').value}`);
      if (mode === 'separate') {
        params.set('reference_width_mm', $('#reference-width').value);
        params.set('reference_height_mm', $('#reference-height').value);
      }
    } else {
      params.set('fit_page', form.elements.fit_page.checked ? 'true' : 'false');
    }
    return params;
  }

  function updateRequest() {
    const endpoint = new URL(form.getAttribute('action'), location.origin);
    const params = options();
    if (isMeasure) {
      endpoint.search = params.toString();
      curl = `curl ${shellQuote(endpoint.href)}`;
    } else {
      const filename = $('#dxf-file').files[0]?.name || 'drawing.dxf';
      // cURL form values use their own quoting for filenames containing punctuation.
      const escapedName = filename.replaceAll('\\', '\\\\').replaceAll('"', '\\"');
      const fields = [`-F ${shellQuote(`${$('#upload-field').value}=@"${escapedName}"`)}`];
      for (const [key, value] of params) fields.push(`--form-string ${shellQuote(`${key}=${value}`)}`);
      curl = `curl ${shellQuote(endpoint.href)} \\\n  ${fields.join(' \\\n  ')}`;
    }
    $('#request-code').textContent = curl;
    $('#request-note').textContent = isMeasure ? 'This request uses the existing measurement API on this server.' : 'The browser uploads your selected file. For cURL, replace the filename with its local path.';
    return {endpoint, params};
  }

  function syncReference() {
    if (!isMeasure) return;
    const custom = ['compact', 'separate'].includes($('#reference-mode').value);
    $('#reference-fields').hidden = !custom;
    for (const input of $('#reference-fields').querySelectorAll('input')) input.disabled = !custom;
  }

  function blobURL(content, type) {
    const url = URL.createObjectURL(new Blob([content], {type}));
    blobs.push(url);
    return url;
  }

  function preview(source, label) {
    const figure = document.createElement('figure');
    const img = document.createElement('img');
    img.src = source;
    img.alt = label;
    img.loading = 'lazy';
    img.addEventListener('error', () => { caption.textContent = `${label} — preview unavailable`; });
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    figure.append(img, caption);
    $('#previews').append(figure);
  }

  function safeImageURL(value) {
    try {
      const url = new URL(value, location.origin);
      return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
    } catch { return null; }
  }

  function svgMarkup(value) {
    if (typeof value !== 'string' || !value.trim()) return null;
    if (value.trim().startsWith('<')) return value;
    // Some engine versions return contour path data instead of a complete SVG.
    const numbers = value.match(/[-+]?(?:\d*\.)?\d+(?:e[-+]?\d+)?/gi)?.map(Number) || [];
    if (numbers.length < 4) return null;
    const xs = numbers.filter((_, index) => index % 2 === 0);
    const ys = numbers.filter((_, index) => index % 2 === 1);
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `${Math.min(...xs)} ${Math.min(...ys)} ${Math.max(...xs) - Math.min(...xs) || 1} ${Math.max(...ys) - Math.min(...ys) || 1}`);
    const path = document.createElementNS(svg.namespaceURI, 'path');
    for (const [key, val] of Object.entries({d: value, fill: 'none', stroke: '#086b62', 'stroke-width': '2', 'vector-effect': 'non-scaling-stroke'})) path.setAttribute(key, val);
    svg.append(path);
    return new XMLSerializer().serializeToString(svg);
  }

  function download(url, name, label) {
    $('#download').href = url;
    $('#download').download = name;
    $('#download').textContent = label;
    $('#download').hidden = false;
  }

  function clearResult() {
    $('#previews').replaceChildren();
    blobs.splice(0).forEach(URL.revokeObjectURL);
    for (const selector of ['#empty', '#summary', '#download', '#response-details']) $(selector).hidden = true;
    $('#response-code').textContent = '';
  }

  form.addEventListener('input', () => {
    if (!isMeasure && document.activeElement === $('#scale')) $('#source-units').value = 'custom';
    syncReference();
    updateRequest();
  });
  form.addEventListener('change', (event) => {
    if (!isMeasure && event.target === $('#source-units') && $('#source-units').value !== 'custom') $('#scale').value = $('#source-units').value;
    syncReference();
    updateRequest();
  });
  form.addEventListener('reset', () => setTimeout(() => {
    syncReference();
    updateRequest();
    if (!busy) {
      clearResult();
      $('#empty').hidden = false;
      $('#status').textContent = 'Ready';
      $('#status').dataset.error = 'false';
    }
  }, 0));
  $('#copy-request').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(curl);
      $('#copy-request').textContent = 'Copied';
      setTimeout(() => { $('#copy-request').textContent = 'Copy cURL'; }, 1800);
    } catch { $('#request-note').textContent = 'Clipboard unavailable. Select and copy the request above.'; }
  });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;
    // Zero is valid for page sizes and margins, but never for scale or reference sizes.
    for (const input of form.querySelectorAll('#scale, #reference-fields input')) {
      input.setCustomValidity(!input.disabled && input.value !== '' && Number(input.value) <= 0 ? 'Enter a number greater than zero.' : '');
    }
    if (!form.reportValidity()) return;
    const {endpoint, params} = updateRequest();
    const body = new FormData();
    if (!isMeasure) {
      body.append($('#upload-field').value, $('#dxf-file').files[0]);
      for (const [key, value] of params) body.append(key, value);
    }
    clearResult();
    busy = true;
    $('#submit').disabled = true;
    form.querySelector('[type=reset]').disabled = true;
    $('#result-panel').setAttribute('aria-busy', 'true');
    $('#status').dataset.error = 'false';
    $('#status').textContent = isMeasure ? 'Measuring…' : 'Converting…';
    const start = performance.now();
    try {
      const response = await fetch(endpoint, isMeasure ? {headers: {Accept: 'application/json'}} : {method: 'POST', body});
      const text = await response.text();
      const json = response.headers.get('Content-Type')?.includes('application/json') ? JSON.parse(text) : null;
      $('#response-code').textContent = json ? JSON.stringify(json, null, 2) : text;
      $('#response-details').hidden = false;
      $('#response-details').open = !response.ok;
      $('#status').textContent = `HTTP ${response.status} · ${((performance.now() - start) / 1000).toFixed(2)} s`;
      $('#status').dataset.error = String(!response.ok);
      if (json) {
        $('#summary').hidden = false;
        $('#summary').textContent = json.error ? [json.error, json.details].filter(Boolean).join('\n') : `Width: ${json.width_mm ?? json.width ?? '—'} mm · Height: ${json.height_mm ?? json.height ?? '—'} mm\nEngine: ${json.measurer ?? '—'} · Objects: ${json.measurements?.length ?? 0}`;
        download(blobURL(text, 'application/json'), 'measurement.json', 'Download JSON');
        const markup = svgMarkup(json.svg || json.debug?.object_contour_svg);
        if (markup) preview(blobURL(markup, 'image/svg+xml'), 'Object contour');
        for (const [name, img] of Object.entries(json.debug_images || {})) {
          if (img.encoding === 'base64' && img.mime_type === 'image/png' && typeof img.data === 'string') preview(`data:image/png;base64,${img.data}`, name);
        }
        for (const img of json.debug_image_urls || []) {
          const url = safeImageURL(img.url);
          if (url) preview(url, img.name || img.filename || 'Debug image');
        }
      } else if (response.ok && response.headers.get('Content-Type')?.includes('image/svg+xml')) {
        const url = blobURL(text, 'image/svg+xml');
        preview(url, 'Converted SVG · scaled to fit this preview');
        download(url, $('#dxf-file').files[0].name.replace(/\.dxf$/i, '') + '.svg', 'Download SVG');
        const root = new DOMParser().parseFromString(text, 'image/svg+xml').documentElement;
        $('#summary').hidden = false;
        $('#summary').textContent = `SVG size: ${root.getAttribute('width')} × ${root.getAttribute('height')}\nviewBox: ${root.getAttribute('viewBox')}\nScale: ${response.headers.get('X-DXF-SVG-Scale')} · Page units: ${response.headers.get('X-DXF-SVG-Units')}`;
      } else {
        $('#summary').hidden = false;
        $('#summary').textContent = 'Unexpected response. Inspect the response body below.';
      }
    } catch (error) {
      $('#status').textContent = 'Request failed';
      $('#status').dataset.error = 'true';
      $('#summary').hidden = false;
      $('#summary').textContent = error.message;
    } finally {
      busy = false;
      $('#submit').disabled = false;
      form.querySelector('[type=reset]').disabled = false;
      $('#result-panel').setAttribute('aria-busy', 'false');
    }
  });
  form.addEventListener('input', (event) => {
    if (event.target instanceof HTMLInputElement) event.target.setCustomValidity('');
  });
  syncReference();
  updateRequest();
})();
