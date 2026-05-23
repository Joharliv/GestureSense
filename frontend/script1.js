const io = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (e.isIntersecting) {
      e.target.classList.add('in');
      io.unobserve(e.target);
    }
  });
}, { threshold: 0.15 });
document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

// Hero caption ticker
const phrases = [
  '"hello, nice to meet you"',
  '"can i get an oat-milk matcha?"',
  '"i\'ll be there in 5"',
  '"that\'s actually wild"',
  '"thank you, seriously"',
];
const ticker = document.getElementById('ticker');
let i = 0;
if (ticker) {
  setInterval(() => {
    i = (i + 1) % phrases.length;
    ticker.style.opacity = '0';
    setTimeout(() => {
      ticker.textContent = phrases[i];
      ticker.style.opacity = '1';
    }, 240);
  }, 2600);
  ticker.style.transition = 'opacity 0.24s ease';
}

// Mobile menu (basic toggle)
const burger = document.querySelector('.nav__burger');
const links = document.querySelector('.nav__links');
if (burger && links) {
  burger.addEventListener('click', () => {
    const open = links.style.display === 'flex';
    links.style.display = open ? 'none' : 'flex';
    links.style.position = 'absolute';
    links.style.top = '64px';
    links.style.left = '0';
    links.style.right = '0';
    links.style.flexDirection = 'column';
    links.style.padding = '20px';
    links.style.background = 'rgba(11,6,18,0.95)';
    links.style.borderBottom = '1px solid rgba(168,85,247,0.22)';
  });
}