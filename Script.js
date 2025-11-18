/* script.js - AJAX + animations for one-page UI */
(() => {
  const form = document.getElementById('stressForm');
  const submitBtn = document.getElementById('submitBtn');
  const demoBtn = document.getElementById('demoBtn');
  const loader = document.getElementById('loader');
  const resultCard = document.getElementById('resultCard');
  const resultContent = document.getElementById('resultContent');
  const resultEmpty = document.getElementById('resultEmpty');

  const stressLabel = document.getElementById('stressLabel');
  const clusterGroup = document.getElementById('clusterGroup');
  const modelAccuracy = document.getElementById('modelAccuracy');
  const adviceBox = document.getElementById('advice');

  const imgs = [
    document.getElementById('imgConf'),
    document.getElementById('imgFeat'),
    document.getElementById('imgKM')
  ];

  // utility: set loading state
  function setLoading(on = true) {
    loader.classList.toggle('hidden', !on);
    if (on) {
      resultContent.classList.add('hidden');
      resultEmpty.classList.add('hidden');
    }
  }

  // animate images when they are loaded
  function animateImages() {
    imgs.forEach(img => {
      if (!img) return;
      if (img.complete) {
        img.classList.add('loaded');
      } else {
        img.onload = () => img.classList.add('loaded');
      }
    });
  }

  // demo filler (example values)
  demoBtn.addEventListener('click', () => {
    form.sleep.value = 7.5;
    form.study.value = 3.5;
    form.exercise.value = 2;
    form.anxiety.value = 2;
    form.depression.value = 1.5;
    form.focus.value = 7;
    form.diet.value = 6;
  });

  // AJAX submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);

    const data = new FormData(form);

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        body: data
      });

      const json = await resp.json();

      if (json.error) throw new Error(json.error || 'Unknown error');

      // fill UI
      const stress = json.stress || 'N/A';
      const cluster = json.cluster ?? '—';
      const accuracy = json.accuracy ?? 'N/A';

      // style badge
      stressLabel.textContent = stress;
      stressLabel.className = 'stress-badge'; // reset classes
      if (stress.toLowerCase().includes('high')) {
        stressLabel.classList.add('stress-high');
        adviceBox.textContent = 'High stress detected — try breathing exercises, breaks, and speak to a counselor if it continues.';
      } else if (stress.toLowerCase().includes('medium')) {
        stressLabel.classList.add('stress-medium');
        adviceBox.textContent = 'Moderate stress — maintain healthy routines and monitor sleep and study balance.';
      } else {
        stressLabel.classList.add('stress-low');
        adviceBox.textContent = 'Low stress — good job! Keep your routines and healthy habits.';
      }

      clusterGroup.innerHTML = `<strong>Cluster:</strong> ${cluster}`;
      modelAccuracy.innerHTML = `${Number(accuracy).toFixed(2)}`;

      // show result area
      setLoading(false);
      resultContent.classList.remove('hidden');
      resultEmpty.classList.add('hidden');
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // refresh graphs by reloading images (to show animations)
      imgs.forEach(img => {
        if (!img) return;
        img.classList.remove('loaded');
        // cache-bust to force browser to re-render (if server regenerates)
        const src = img.getAttribute('src').split('?')[0];
        img.src = src + '?v=' + Date.now();
      });

      // animate after small delay
      setTimeout(animateImages, 250);

    } catch (err) {
      setLoading(false);
      resultEmpty.querySelector('p').textContent = 'Error: ' + (err.message || err);
      resultEmpty.classList.remove('hidden');
    }
  });

  // Reset button
  document.getElementById('resetBtn').addEventListener('click', () => {
    form.reset();
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    resultContent.classList.add('hidden');
    resultEmpty.classList.remove('hidden');
    imgs.forEach(img => img.classList.remove('loaded'));
  });

  // initial preload animate check
  window.addEventListener('load', () => {
    imgs.forEach(img => {
      if (img && img.complete) img.classList.add('loaded');
    });
  });
})();
