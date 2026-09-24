/**
 * LoanGuard AI - Frontend Controller & Model Client
 * Week 4 Machine Learning Project
 */

// Determine API Base URL
const API_BASE = (window.location.origin.includes(':5000') || window.location.pathname.startsWith('/'))
  ? '' 
  : 'http://127.0.0.1:5000';

// Fallback Presets
const PRESETS = {
  low_risk: {
    Age: 52,
    Education: "PhD",
    MaritalStatus: "Married",
    HasDependents: "Yes",
    Income: 135000,
    EmploymentType: "Full-time",
    MonthsEmployed: 108,
    CreditScore: 795,
    NumCreditLines: 2,
    DTIRatio: 0.18,
    LoanAmount: 45000,
    InterestRate: 6.8,
    LoanTerm: 24,
    LoanPurpose: "Home",
    HasMortgage: "Yes",
    HasCoSigner: "Yes"
  },
  high_risk: {
    Age: 22,
    Education: "High School",
    MaritalStatus: "Single",
    HasDependents: "No",
    Income: 24000,
    EmploymentType: "Unemployed",
    MonthsEmployed: 6,
    CreditScore: 430,
    NumCreditLines: 4,
    DTIRatio: 0.78,
    LoanAmount: 185000,
    InterestRate: 22.8,
    LoanTerm: 60,
    LoanPurpose: "Business",
    HasMortgage: "No",
    HasCoSigner: "No"
  },
  average: {
    Age: 42,
    Education: "Bachelor's",
    MaritalStatus: "Married",
    HasDependents: "No",
    Income: 82500,
    EmploymentType: "Full-time",
    MonthsEmployed: 60,
    CreditScore: 650,
    NumCreditLines: 2,
    DTIRatio: 0.40,
    LoanAmount: 120000,
    InterestRate: 12.5,
    LoanTerm: 36,
    LoanPurpose: "Home",
    HasMortgage: "Yes",
    HasCoSigner: "Yes"
  }
};

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSliders();
  initFormControls();
  checkBackendHealth();
});

// ==========================================================================
// 1. Tab Navigation
// ==========================================================================
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      
      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const activePane = document.getElementById(targetId);
      if (activePane) {
        activePane.classList.add('active');
      }
    });
  });
}

// ==========================================================================
// 2. Interactive Sliders & Live Labels
// ==========================================================================
function initSliders() {
  const monthsSlider = document.getElementById('input-months-employed');
  const monthsVal = document.getElementById('val-months-employed');
  const yearsVal = document.getElementById('val-years-employed');

  if (monthsSlider && monthsVal) {
    monthsSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      monthsVal.textContent = val;
      if (yearsVal) {
        yearsVal.textContent = (val / 12).toFixed(1);
      }
    });
  }

  const dtiSlider = document.getElementById('input-dti');
  const dtiVal = document.getElementById('val-dti');
  const dtiPct = document.getElementById('val-dti-pct');

  if (dtiSlider && dtiVal) {
    dtiSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      dtiVal.textContent = val.toFixed(2);
      if (dtiPct) {
        dtiPct.textContent = Math.round(val * 100) + '%';
      }
    });
  }

  const creditInput = document.getElementById('input-credit-score');
  const creditVal = document.getElementById('val-credit-score');
  const creditTier = document.getElementById('badge-credit-tier');

  if (creditInput) {
    creditInput.addEventListener('input', (e) => {
      const score = parseInt(e.target.value) || 300;
      if (creditVal) creditVal.textContent = score;
      
      if (creditTier) {
        if (score >= 750) {
          creditTier.textContent = `Excellent (${score})`;
          creditTier.style.color = 'var(--success)';
        } else if (score >= 670) {
          creditTier.textContent = `Good (${score})`;
          creditTier.style.color = '#60a5fa';
        } else if (score >= 580) {
          creditTier.textContent = `Fair (${score})`;
          creditTier.style.color = 'var(--warning)';
        } else {
          creditTier.textContent = `Poor (${score})`;
          creditTier.style.color = 'var(--danger)';
        }
      }
    });
  }
}

// ==========================================================================
// 3. Form & Action Controls
// ==========================================================================
function initFormControls() {
  const form = document.getElementById('loan-form');
  const btnRandom = document.getElementById('btn-random-values');
  const btnLow = document.getElementById('btn-preset-low');
  const btnHigh = document.getElementById('btn-preset-high');
  const btnAvg = document.getElementById('btn-preset-avg');
  const btnReset = document.getElementById('btn-reset-form');

  // Submit
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      await handlePrediction();
    });
  }

  // Random Values Button
  if (btnRandom) {
    btnRandom.addEventListener('click', async () => {
      animateButton(btnRandom);
      await fetchOrGenerateRandom('random');
    });
  }

  // Presets
  if (btnLow) btnLow.addEventListener('click', () => applyPreset('low_risk'));
  if (btnHigh) btnHigh.addEventListener('click', () => applyPreset('high_risk'));
  if (btnAvg) btnAvg.addEventListener('click', () => applyPreset('average'));

  // Reset
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      applyPreset('average');
      resetResultCard();
    });
  }
}

function animateButton(btn) {
  btn.style.transform = 'scale(0.95)';
  setTimeout(() => { btn.style.transform = ''; }, 150);
}

// ==========================================================================
// 4. Fill Random & Preset Values
// ==========================================================================
async function fetchOrGenerateRandom(mode = 'random') {
  try {
    const res = await fetch(`${API_BASE}/api/random-sample?mode=${mode}`);
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'success' && data.sample) {
        populateForm(data.sample);
        return;
      }
    }
  } catch (err) {
    console.log('[INFO] Backend random API not reachable, generating client-side random values.');
  }

  // Client-side fallback randomizer
  const educations = ["Bachelor's", "High School", "Master's", "PhD"];
  const employments = ["Full-time", "Part-time", "Self-employed", "Unemployed"];
  const maritals = ["Married", "Single", "Divorced"];
  const purposes = ["Home", "Business", "Auto", "Education", "Other"];
  const yesNo = ["No", "Yes"];

  const randomSample = {
    Age: Math.floor(Math.random() * (65 - 21 + 1)) + 21,
    Income: Math.floor(Math.random() * (145000 - 25000) / 1000) * 1000 + 25000,
    LoanAmount: Math.floor(Math.random() * (240000 - 15000) / 1000) * 1000 + 15000,
    CreditScore: Math.floor(Math.random() * (820 - 350)) + 350,
    MonthsEmployed: Math.floor(Math.random() * 115),
    NumCreditLines: Math.floor(Math.random() * 4) + 1,
    InterestRate: parseFloat((Math.random() * (24.5 - 4.5) + 4.5).toFixed(1)),
    LoanTerm: [12, 24, 36, 48, 60][Math.floor(Math.random() * 5)],
    DTIRatio: parseFloat((Math.random() * (0.85 - 0.15) + 0.15).toFixed(2)),
    Education: educations[Math.floor(Math.random() * educations.length)],
    EmploymentType: employments[Math.floor(Math.random() * employments.length)],
    MaritalStatus: maritals[Math.floor(Math.random() * maritals.length)],
    HasMortgage: yesNo[Math.floor(Math.random() * yesNo.length)],
    HasDependents: yesNo[Math.floor(Math.random() * yesNo.length)],
    LoanPurpose: purposes[Math.floor(Math.random() * purposes.length)],
    HasCoSigner: yesNo[Math.floor(Math.random() * yesNo.length)]
  };

  populateForm(randomSample);
}

function applyPreset(presetKey) {
  const preset = PRESETS[presetKey] || PRESETS.average;
  populateForm(preset);
}

function populateForm(data) {
  const form = document.getElementById('loan-form');
  if (!form) return;

  for (const [key, val] of Object.entries(data)) {
    const input = form.elements[key];
    if (input) {
      input.value = val;
      // Trigger input event to update dynamic labels & sliders
      input.dispatchEvent(new Event('input'));
    }
  }
}

// ==========================================================================
// 5. Prediction Execution
// ==========================================================================
async function handlePrediction() {
  const form = document.getElementById('loan-form');
  if (!form) return;

  // Extract form values
  const formData = new FormData(form);
  const payload = {};
  formData.forEach((val, key) => {
    payload[key] = val;
  });

  // UI: Show loading
  const placeholder = document.getElementById('result-placeholder');
  const loading = document.getElementById('result-loading');
  const content = document.getElementById('result-content');

  if (placeholder) placeholder.style.display = 'none';
  if (content) content.style.display = 'none';
  if (loading) loading.style.display = 'block';

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const result = await res.json();
      renderPrediction(result);
      return;
    }
  } catch (err) {
    console.log('[INFO] Backend prediction endpoint not reached. Running calibrated client-side engine.');
  }

  // Client-side fallback prediction if server is offline
  setTimeout(() => {
    const fallbackResult = runClientSideInference(payload);
    renderPrediction(fallbackResult);
  }, 400);
}

function renderPrediction(res) {
  const loading = document.getElementById('result-loading');
  const content = document.getElementById('result-content');
  if (loading) loading.style.display = 'none';
  if (content) content.style.display = 'block';

  const isDefault = res.prediction === 1;
  const probPct = res.default_probability_pct;
  const riskTier = res.risk_tier;

  // 1. Verdict Banner
  const banner = document.getElementById('verdict-banner');
  const icon = document.getElementById('verdict-icon');
  const tier = document.getElementById('verdict-tier');
  const title = document.getElementById('verdict-title');
  const desc = document.getElementById('verdict-desc');

  banner.className = 'verdict-banner';
  if (riskTier === 'Low Risk') {
    banner.classList.add('verdict-low');
    icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
  } else if (riskTier === 'Moderate Risk') {
    banner.classList.add('verdict-moderate');
    icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
  } else {
    banner.classList.add('verdict-high');
    icon.innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
  }

  if (tier) tier.textContent = riskTier.toUpperCase();
  if (title) title.textContent = res.decision;
  if (desc) desc.textContent = res.decision_subtext;

  // 2. Probability Gauge
  const probNum = document.getElementById('prob-number');
  const probFill = document.getElementById('prob-bar-fill');

  if (probNum) probNum.textContent = `${probPct}%`;
  if (probFill) {
    probFill.style.width = `${probPct}%`;
    if (probPct < 35) {
      probFill.style.background = 'linear-gradient(90deg, #059669, #10b981)';
    } else if (probPct < 60) {
      probFill.style.background = 'linear-gradient(90deg, #d97706, #f59e0b)';
    } else {
      probFill.style.background = 'linear-gradient(90deg, #dc2626, #ef4444)';
    }
  }

  // 3. Financial Summary
  const fin = res.financial_summary || {};
  const emiEl = document.getElementById('fin-emi');
  const totalEl = document.getElementById('fin-total');
  const interestEl = document.getElementById('fin-interest');
  const burdenEl = document.getElementById('fin-burden');

  if (emiEl) emiEl.textContent = formatCurrency(fin.monthly_emi);
  if (totalEl) totalEl.textContent = formatCurrency(fin.total_repayment);
  if (interestEl) interestEl.textContent = formatCurrency(fin.total_interest);
  if (burdenEl) burdenEl.textContent = `${fin.debt_burden_pct}% of Income`;

  // 4. Contributing Factors
  const factorsList = document.getElementById('factors-list');
  if (factorsList) {
    factorsList.innerHTML = '';
    const factors = res.risk_factors || [];

    if (factors.length === 0) {
      factorsList.innerHTML = '<li class="factor-item factor-neutral"><i class="fa-solid fa-check"></i> Standard baseline financial indicators.</li>';
    } else {
      factors.forEach(f => {
        const li = document.createElement('li');
        li.className = `factor-item factor-${f.type}`;
        const iconClass = f.type === 'positive' 
          ? 'fa-circle-check' 
          : (f.type === 'negative' ? 'fa-triangle-exclamation' : 'fa-circle-info');
        li.innerHTML = `<i class="fa-solid ${iconClass}"></i> <span>${f.text}</span>`;
        factorsList.appendChild(li);
      });
    }
  }

  // 5. Engine text
  const engineEl = document.getElementById('inference-engine-text');
  if (engineEl) {
    engineEl.textContent = res.inference_engine || 'HistGradientBoostingClassifier Pipeline';
  }
}

function resetResultCard() {
  const placeholder = document.getElementById('result-placeholder');
  const loading = document.getElementById('result-loading');
  const content = document.getElementById('result-content');

  if (placeholder) placeholder.style.display = 'block';
  if (loading) loading.style.display = 'none';
  if (content) content.style.display = 'none';
}

function formatCurrency(num) {
  if (isNaN(num)) return '$0';
  return '$' + Math.round(num).toLocaleString();
}

// ==========================================================================
// 6. Client-Side Inference Engine (Guaranteed Zero-Downtime Fallback)
// ==========================================================================
function runClientSideInference(data) {
  const age = parseFloat(data.Age) || 42;
  const income = parseFloat(data.Income) || 82500;
  const loanAmount = parseFloat(data.LoanAmount) || 120000;
  const creditScore = parseFloat(data.CreditScore) || 650;
  const monthsEmployed = parseFloat(data.MonthsEmployed) || 60;
  const numCreditLines = parseFloat(data.NumCreditLines) || 2;
  const interestRate = parseFloat(data.InterestRate) || 12.5;
  const loanTerm = parseInt(data.LoanTerm) || 36;
  const dti = parseFloat(data.DTIRatio) || 0.40;

  // Calibrated logistic regression weights from full 255k dataset
  let z = -1.25;
  z += ((age - 43.5) / 15.0) * (-0.60);
  z += ((income - 82500) / 39000) * (-0.30);
  z += ((loanAmount - 127500) / 70800) * (0.31);
  z += ((creditScore - 574) / 159) * (-0.13);
  z += ((monthsEmployed - 59.5) / 34.6) * (-0.37);
  z += ((numCreditLines - 2.5) / 1.1) * (0.10);
  z += ((interestRate - 13.5) / 6.6) * (0.49);
  z += ((loanTerm - 36) / 17.0) * (-0.01);
  z += ((dti - 0.50) / 0.23) * (0.08);

  if (data.Education === 'High School') z += 0.04;
  if (data.Education === "Master's") z -= 0.07;
  if (data.Education === 'PhD') z -= 0.11;

  if (data.EmploymentType === 'Part-time') z += 0.10;
  if (data.EmploymentType === 'Self-employed') z += 0.08;
  if (data.EmploymentType === 'Unemployed') z += 0.25;

  if (data.MaritalStatus === 'Married') z -= 0.11;
  if (data.HasMortgage === 'Yes') z -= 0.10;
  if (data.HasDependents === 'Yes') z -= 0.13;
  if (data.HasCoSigner === 'Yes') z -= 0.12;

  let prob = 1.0 / (1.0 + Math.exp(-z));
  prob = Math.max(0.02, Math.min(0.98, prob));
  const probPct = Math.round(prob * 1000) / 10;
  const prediction = prob >= 0.50 ? 1 : 0;

  let riskTier = "Low Risk";
  let decision = "Approval Recommended";
  let decisionSubtext = "Applicant demonstrates strong financial indicators and low default probability.";

  if (prob < 0.35) {
    riskTier = "Low Risk";
    decision = "Approval Recommended";
    decisionSubtext = "Applicant demonstrates strong financial indicators and low default probability.";
  } else if (prob < 0.60) {
    riskTier = "Moderate Risk";
    decision = "Conditional Approval / Manual Review";
    decisionSubtext = "Applicant presents mixed credit indicators. Collateral or co-signer strongly advised.";
  } else {
    riskTier = "High Risk";
    decision = "High Default Risk / Rejection Advised";
    decisionSubtext = "Applicant profile exhibits elevated risk factors that breach safe default thresholds.";
  }

  // EMI calculation
  const r = (interestRate / 100.0) / 12.0;
  const n = loanTerm;
  const P = loanAmount;
  let emi = 0;
  if (r > 0 && n > 0) {
    emi = P * (r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
  } else {
    emi = P / n;
  }
  const totalRepayment = emi * n;
  const totalInterest = Math.max(0, totalRepayment - P);
  const monthlyIncome = Math.max(1.0, income / 12.0);
  const debtBurdenPct = Math.round((emi / monthlyIncome) * 1000) / 10;

  // Factors
  const factors = [];
  if (dti >= 0.55) factors.push({ type: "negative", text: `High Debt-to-Income ratio (${dti.toFixed(2)}) leaves narrow cash buffer.` });
  else if (dti <= 0.25) factors.push({ type: "positive", text: `Low Debt-to-Income ratio (${dti.toFixed(2)}) indicates comfortable payment capacity.` });

  if (creditScore < 550) factors.push({ type: "negative", text: `Low credit score (${creditScore}) flags heightened credit delinquency risk.` });
  else if (creditScore >= 720) factors.push({ type: "positive", text: `High credit score (${creditScore}) reflects excellent repayment history.` });

  if (interestRate >= 18.0) factors.push({ type: "negative", text: `Elevated interest rate (${interestRate.toFixed(1)}%) increases monthly debt pressure.` });
  else if (interestRate <= 9.0) factors.push({ type: "positive", text: `Competitive interest rate (${interestRate.toFixed(1)}%) minimizes borrowing costs.` });

  if (data.EmploymentType === 'Unemployed') factors.push({ type: "negative", text: `Unemployed status poses immediate cash-flow risk.` });
  else if (monthsEmployed >= 48) factors.push({ type: "positive", text: `Long employment tenure (${Math.floor(monthsEmployed/12)}+ years) ensures steady income.` });

  if (data.HasCoSigner === 'Yes') factors.push({ type: "positive", text: `Co-Signer presence provides secondary financial guarantee.` });

  return {
    status: "success",
    prediction: prediction,
    prediction_label: prediction === 1 ? "Default Predicted" : "Non-Default (Will Pay)",
    default_probability: Math.round(prob * 10000) / 10000,
    default_probability_pct: probPct,
    risk_tier: riskTier,
    decision: decision,
    decision_subtext: decisionSubtext,
    financial_summary: {
      monthly_emi: Math.round(emi),
      total_repayment: Math.round(totalRepayment),
      total_interest: Math.round(totalInterest),
      debt_burden_pct: debtBurdenPct
    },
    risk_factors: factors,
    inference_engine: "Calibrated Balanced ML Engine (Client Fallback)"
  };
}

// ==========================================================================
// 7. Backend Healthcheck
// ==========================================================================
async function checkBackendHealth() {
  const statusWrap = document.getElementById('server-status');
  const statusText = document.getElementById('status-text');

  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      if (statusWrap) {
        statusWrap.className = 'server-status connected';
        statusText.textContent = data.model_loaded ? 'Model Live 🟢' : 'API Online 🟢';
      }
      return;
    }
  } catch (err) {
    // offline
  }

  if (statusWrap) {
    statusWrap.className = 'server-status error';
    statusText.textContent = 'Offline Mode 🟡';
  }
}
