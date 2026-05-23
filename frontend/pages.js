(function () {
  const wrap = document.getElementById('particles');
  if (!wrap) return;
  const count = window.innerWidth < 720 ? 18 : 36;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('span');
    p.className = 'particle';
    const size = Math.random() * 4 + 2;
    p.style.width = p.style.height = `${size}px`;
    p.style.left = `${Math.random() * 100}%`;
    p.style.top = `${100 + Math.random() * 20}%`;
    p.style.animationDuration = `${12 + Math.random() * 18}s`;
    p.style.animationDelay = `${-Math.random() * 20}s`;
    p.style.opacity = String(0.3 + Math.random() * 0.5);
    wrap.appendChild(p);
  }
})();
// Contact form fake submit
(function () {
  const form = document.getElementById('contactForm');
  const status = document.getElementById('formStatus');
  if (!form) return;
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    status.textContent = 'Sending…';
    setTimeout(() => {
      status.textContent = '✓ Thanks — we\'ll be in touch within 48h.';
      form.reset();
    }, 700);
  });
})();