# General Spot Trading Agent — المرجع الشامل / Complete Reference

> **وثيقة مرجعية كاملة للنظام**: كيف يعمل النموذج، كيف تتحكم بالتدريب من التيرمينال، كيف تعرف أنه انتهى، وكيف تستخدمه وتربطه بأي منصة أو نظام.
>
> **ملحوظة عن اللغة**: الأقسام التقنية (الأوامر، أسماء الملفات، المصطلحات) تُكتب بالإنجليزية لأنها هي التي ستكتبها في التيرمينال والكود. الشرح حولها بالعربية. هذا يمنع مشاكل المحاذاة والاتجاه (RTL/LTR mix) التي تظهر عند خلط النصوص داخل سطر واحد.

---

## 1. نظرة عامة على النظام | System Overview

نظام تداول Spot يعتمد على التعلم المعزز (Reinforcement Learning) بنموذج PPO واحد عام يتعلم من بيانات عشرات العملات الرقمية. النموذج لا يعرف اسم العملة التي يراها — بل يتعلم أنماط السوق العامة نفسها.

### البنية المعمارية | Architecture

```
Model → Harness → Safety → Execution → Exchange
```

| الطبقة | المجلد | الوظيفة |
|---|---|---|
| Model | `src/model/` | شبكة GRU + Beta policy: تقرأ نافذة 96 شمعة وتقرر التخصيص |
| Features | `src/features/` | 92 ميزة لكل شمعة (Elliott/Wyckoff/Gann/MTF/cycles كإشارات وليس قواعد) |
| Training | `src/training/` | بيئة التداول + PPO مع رسوم وانزلاق محسوب |
| Dataset | `src/dataset.py` | تقسيم زمني 70/15/15 لكل عملة + scaler على train فقط |
| Evaluation | `src/evaluation/` | تقييم على validation/test/unseen + مقارنة buy&hold |
| Harness | `src/harness/` | طبقة إدارة التشغيل (حالة، دورة، قرار) |
| Safety | `src/execution/safety.py` | حدود الأمان (حجم الأمر، حد الخسارة اليومي) |
| Execution | `src/execution/` | CCXT adapter للربط مع أي منصة CEX |
| MCP | `src/mcp/` | واجهة تحكم ومراقبة عبر بروتوكول MCP |

### قرارات تصميمية أساسية | Core Decisions

- **نموذج واحد عام**: كل العملات مصادر تدريب (episodes) — وليست محفظة يتداولها معاً
- **Spot فقط**: لا futures، لا margin، لا leverage، لا short
- **الهدف = قيمة المحفظة**: المكافأة هي log التغير في القيمة بعد الرسوم والانزلاق
- **حالة المحفظة في الملاحظة**: النموذج يرى (position, pnl, drawdown, time) فيتعلم إدارة المخاطر
- **Deadband**: صفقات أقل من 2% من المحفظة تُتجاهل — يمنع أكل الرسوم
- **عملات unseen**: ADA/DOT/LTC محجوزة كلياً لاختبار قدرة النموذج على التعميم

---

## 2. البيانات | Data

**العملات الـ 20 المحمّلة** (فريم 1h، من 2020 حتى اليوم):

| مجموعة | العملات | ملاحظات |
|---|---|---|
| Majors | BTC, ETH, BNB, XRP, DOGE, LTC | تاريخ كامل من 2020 |
| L1s | SOL, AVAX, ATOM, NEAR, ALGO | من 2020-2021 |
| Infra | LINK, INJ, APT, ARB, OP, SUI, ICP | ICP بدأ من 2022-01 (حذفنا أول 8 أشهر) |
| **Unseen** | **ADA, DOT, LTC** | لا يراها النموذج أبداً — لاختبار التعميم |

- الملفات الخام: `data/raw/` (61MB) — الميزات المعالجة: `data/processed/` (~1GB)
- كل عملة مقسومة زمنياً: 70% train / 15% validation / 15% test (بدون تداخل)
- الـ scaler مُدرّب على train فقط — لا تسريب من المستقبل

**لإضافة عملات جديدة** (النموذج يظل واحداً — فقط مصادر episodes أكثر):

```bash
.venv/bin/python -m src.data download --symbols NEW/USDT --timeframe 1h --since 2020-01-01
.venv/bin/python -m src.dataset --raw-dir data/raw --processed-dir data/processed --unseen-symbols ADA/USDT DOT/USDT LTC/USDT
```

---

## 3. التحكم بالتدريب | Training Control

### سكربت التحكم الكامل | The Control Script

```bash
./trainctl.sh start          # ابدأ التدريب (المدة من config: 8 ساعات)
./trainctl.sh start 6        # أو حدد المدة يدوياً
./trainctl.sh stop           # إيقاف آمن: ينهي الخطوة الحالية، يحفظ، يخرج (~دقيقة)
./trainctl.sh resume         # استئناف من آخر checkpoint
./trainctl.sh resume 6       # استئناف بمدة محددة
./trainctl.sh status         # الحالة: يعمل؟ + GPU + آخر تقدم + آخر checkpoint
./trainctl.sh watch          # متابعة حية (Ctrl+C يخرج فقط، لا يوقف التدريب)
./trainctl.sh log 20          # آخر 20 سطر تقدم
./trainctl.sh kill           # قتل فوري (طوارئ فقط — resume يعمل بعده عادي)
./trainctl.sh help           # مساعدة
```

### الأوامر المباشرة | Direct Commands

```bash
.venv/bin/python -m src.train --help              # كل الخيارات
.venv/bin/python -m src.train --config config/default.json --time-budget-hours 6
.venv/bin/python -m src.train --resume models/latest.pt --time-budget-hours 12
```

### كيف تبدو جلسة التدريب | Session Anatomy

```
update 350/10000 portfolio=11234 val=8.41 best=9.12% ent=-0.089 kl=0.0041 3.1s
```

| الحقل | المعنى | الصحي |
|---|---|---|
| `update` | رقم خطوة التدريب | يتقدم |
| `portfolio` | متوسط قيمة المحفظة في حلقات التدريب (10K بداية) | يصعد مع الوقت |
| `val` | متوسط العائد % على بيانات validation (كل 25 update) | يصعد ويتجاوز buy-hold |
| `best` | أفضل val مسجل (يُحفظ له best.pt) | يصعد باستمرار |
| `ent` | entropy — كم يستكشف النموذج | لا ينهار نحو أقصى السالب |
| `kl` | انحراف التحديث عن السلوك القديم | تحت 0.02 غالباً |
| `3.1s` | زمن الخطوة | مستقر ~3s على 4060 |

### كيف تعرف أن التدريب انتهى | How to Know It's Done

التدريب يتوقف تلقائياً في واحدة من ثلاث حالات:

1. **انتهاء الوقت**: `time budget of Xh reached at update N` — وصل المحدد بـ `start 6` أو config
2. **Early stopping**: `early stop: no val improvement in N updates` — validation لم يتحسن 400 update متتالية (معناه النموذج وصل لذروته — لا داعي لمزيد)
3. **اكتمال الهدف**: `update 10000/10000` — استنفد عدد الخطوات الأقصى

**للتأكد من التوقف**: `./trainctl.sh status` تظهر `NOT RUNNING` + آخر checkpoint.

**بعد أي توقف**: النموذج جاهز للتقييم، أو استئناف تدريب إضافي anytime بـ `resume`.

### النمط اليومي المقترح | Daily Pattern

```bash
# قبل النوم:
./trainctl.sh start 12
# الصبح (أو توقف وحده بعد 12 ساعة):
./trainctl.sh status
./trainctl.sh resume 6     # قبل العمل، يكمل من حيث وقف
```

---

## 4. التقييم | Evaluation

بعد انتهاء التدريب (أو أثناءه — كل checkpoint قابل للتقييم):

```bash
# على validation (نفس البيانات التي اختير عليها best.pt)
.venv/bin/python -m src.evaluation.evaluate --checkpoint models/best.pt --split validation

# على test (لم يرها النموذج نهائياً أثناء التدريب — الرقم الأصدق)
.venv/bin/python -m src.evaluation.evaluate --checkpoint models/best.pt --split test

# على unseen assets — عملات لم يرها نهائياً (اختبار التعميم الحقيقي)
.venv/bin/python -m src.evaluation.evaluate --checkpoint models/best.pt --split unseen_test
```

**كيف تقرأ النتائج | Reading Results:**

```
BTC/USDT {'final': 11450.7, 'return_pct': 14.5, 'max_drawdown': -18.3, 'trades': 42,
          'fees': 89.2, 'buy_hold': 11800.0, 'beat_buyhold': False}
```

- `return_pct`: عائد الاستراتيجية على كامل فترة test
- `max_drawdown`: أعمق خسارة من القمة (للنسبة المخاطرة)
- `beat_buyhold`: هل تفوق على الشراء والاحتفاظ؟ — هذا هو المقياس الحقيقي
- النموذج ناجح إذا: متوسط return_pct موجب على test + نسبة beat_buyhold جيدة على unseen

**مقارنة checkpoints | A/B Compare:**

```bash
# لو أردت المقارنة بين عدة checkpoints:
for ck in models/update_00100.pt models/update_00500.pt models/best.pt; do
  echo "=== $ck ==="
  .venv/bin/python -m src.evaluation.evaluate --checkpoint $ck --split validation
done
```

---

## 5. استخدام النموذج بعد التدريب | Using the Trained Model

### هيكل الـ Checkpoint | Checkpoint Anatomy

كل ملف `.pt` في `models/` يحتوي:

| المفتاح | المحتوى |
|---|---|
| `model` | أوزان الشبكة |
| `optimizer` | حالة الـ optimizer (للاستئناف) |
| `update`, `best_val` | تقدم التدريب |
| `config` | كل الإعدادات المستخدمة |
| `feature_count` | عدد الميزات (92) — مهم للأحمال الخارجية |
| `state_dim` | أبعاد المدخل الكاملة (92+4=96) |

### أقل مثال تحميل | Minimal Loading Example

```python
import torch
from src.model import TradingAgent

ck = torch.load("models/best.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]
model = TradingAgent(ck["state_dim"], cfg["encoder_size"], cfg["hidden_size"], cfg["num_gru_layers"])
model.load_state_dict(ck["model"]); model.eval()

# window: آخر 96 شمعة بصيغة [96, 96] (92 ميزة + 4 حالة محفظة)
# hidden: يمرر بين النداءات المتتالية للذاكرة
from src.harness.decision import DecisionEngine
engine = DecisionEngine(model, torch.device("cpu"))
result, hidden = engine.predict(window)   # → {"target_position": 0.0..1.0, "value": ...}
```

**تحذير**: الميزات يجب أن تُحسب بنفس `FeatureEngine` ثم تُطبَّع بنفس `scaler.joblib` من `data/processed/` — وإلا فالمدخلات غير صالحة:

```python
import joblib, pandas as pd
from src.features import FeatureEngine

scaler = joblib.load("data/processed/scaler.joblib")
feats = FeatureEngine().transform(latest_200_candles_df)      # احسب الميزات
z = scaler.transform(feats.values)                            # طبّع
window = z[-96:].astype("float32")                            # آخر 96 شمعة
```

### مسار الإنتاج الكامل | Production Path

```
Training → Validation → Unseen Assets → Backtest → Paper Trading → Live Spot
```

الحالة الحالية: كل شيء جاهز حتى Backtest. الباقي مكوّنات موجودة (harness/safety/execution) لكنها بذور تحتاج تعبئة عند الوصول لمرحلتها — بالترتيب، ولواحقة بمنصات حقيقية لاحقاً.

---

## 6. الربط مع المنصات والأنظمة | Integration

### واجهة MCP (OpenCode وأي عميل MCP)

أضف لـ `___open000code.jsonc` (أو `~/.config/opencode/opencode.json`):

```jsonc
{
  "mcp": {
    "servers": {
      "trading-agent": {
        "type": "local",
        "command": [".venv/bin/python", "-m", "src.mcp.server"],
        "cwd": "/home/alsharif/Documents/Programming/trading-agent-final",
        "timeout": 5000
      }
    }
  }
}
```

الأدوات المتاحة الآن: `get_status`, `pause_trading`, `emergency_stop`. (سيتم توسيعها لاحقاً لتشمل مراقبة التدريب والتقييم).

**مهم**: OpenCode واجهة مراقبة/تحكم فقط — النظام يعمل مستقلاً 24/7 بدونه تماماً.

### الربط مع أي منصة CEX عبر CCXT

الطبقة موجودة في `src/execution/`:
- `ExchangeAdapter`: واجهة موحدة (balance, ticker, target_position, cancel_all)
- `CCXTSpotAdapter`: تنفيذ Binance (قابل للتوسيع لأي منصة يدعمها CCXT: Bybit, OKX, Coinbase...)
- `OrderManager`: يخطط الصفقات بعد مرور `SafetyLayer` (حدود حجم الأمر، حد الخسارة اليومي)

Live execution معطل افتراضياً — يُفعّل صراحةً بتمرير `enable_live=True` بعد إتمام مراحل paper-trading والتحقق.

### الربط مع DEX (مستقبلي)

نفس نمط `ExchangeAdapter` — adapter جديد لمنصات DEX (مثلاً عبر APIs خاصة أو web3). طبقة Safety تبقى نفسها لأنها فوق الـ adapter مباشرة.

### تضمين النموذج في أي نظام Python آخر

النموذج يعتمد على torch فقط وقت الاستدلال. حمّل الـ checkpoint كما في المثال أعلاه، زوّد `predict(window, hidden)` كل ساعة، وترجم `target_position` إلى أمر شراء/بيع عبر `OrderManager`. لا حاجة للمشروع كله — يكفي: `src/model/`, `src/features/`, `src/harness/decision.py`, `scaler.joblib`, والـ checkpoint.

---

## 7. الملفات والقراءة | Files & Reading Logs

| الملف | المحتوى |
|---|---|
| `models/best.pt` | أفضل نموذج حسب validation — **هذا الذي تستخدمه** |
| `models/latest.pt` | آخر حالة (للاستئناف) |
| `models/update_XXXXX.pt` | لقطات دورية كل 500 update (للمقارنات) |
| `logs/metrics.csv` | كل سطر تقدم + نتائج validation (افتحه بأي جدول) |
| `logs/train.log` | سجل التدريب الكامل |
| `logs/STOP` | ملف علم الإيقاف الآمن (أنشئه بـ `touch` ليفهم `stop`) |
| `data/processed/manifests` | خرائط البيانات لكل split |

**متابعة تقدم validation عبر الزمن:**

```bash
# ارسم أعمدة val من metrics.csv (فقط الأسطر التي فيها تقييم):
grep -v ',,' logs/metrics.csv | column -t -s,
```

---

## 8. استكشاف الأخطاء | Troubleshooting

| المشكلة | الحل |
|---|---|
| `./trainctl.sh stop` لا يستجب فوراً | طبيعي — ينتظر انتهاء الخطوة الحالية (~3s) + حفظ. تحقق بعد دقيقة |
| انطفأ الجهاز أثناء التدريب | لا مشكلة — `./trainctl.sh resume`. أحدث checkpoint قبل 5 دقائق كحد أقصى |
| `CUDA out of memory` | قلل `num_envs` في config إلى 16 أو 8 |
| بطء شديد بعد ساعات | تحقق `nvidia-smi` — قد تكون عملية أخرى تأخذ GPU. استخدم `./trainctl.sh kill` ثم `resume` |
| DNS/شبكة انقطعت أثناء تنزيل بيانات | أعد نفس أمر التنزيل — يكمل من الكاش |
| `val=-` دائماً في البداية | طبيعي — أول تقييم عند update 25 |
| validation لا يتحسن | راجع قسم 4 — قد تحتاج بيانات أكثر أو تقييم عدة checkpoints قديمة |

---

## 9. الإعدادات | Configuration Reference

`config/default.json` — أهم المفاتيح:

| المفتاح | الافتراضي | المعنى |
|---|---|---|
| `num_envs` | 32 | عدد البيئات المتوازية (قيس على جهازك: 32 = أفضل إنتاجية/8GB VRAM) |
| `rollout_steps` | 512 | طول الحلقة قبل كل تحديث PPO |
| `episode_steps` | 512 | طول حلقة التداول بالشموع |
| `updates` | 10000 | حد أقصى لعدد التحديثات (كافٍ لـ12h+) |
| `time_budget_hours` | 8.0 | ساعات الجلسة الافتراضية |
| `eval_every` | 25 | كل كم update يعمل validation |
| `patience_updates` | 400 | early stopping إذا لم يتحسن |
| `max_kl` | 0.02 | حارس انفجار التحديث |
| `fee` / `slippage` | 0.001 / 0.0005 | رسوم وانزلاق محسوبان في البيئة والتقييم |
| `window` | 96 | كم شمعة يرى النموذج (96 = 4 أيام) |
| `minibatch_size` | 4 | حجم minibatch بيئات (GRU يرى التسلسل كاملاً) |

---

## 10. الاختبارات | Tests

```bash
.venv/bin/python -m pytest tests/ -q     # كل الاختبارات (9)
.venv/bin/python tests/test_system.py    # أو مباشرة
```

تغطي: عدم تسريب المستقبل في الميزات (prefix-invariance)، حارس النافذة، اتساق محور الزمن بين forward وforward_sequence (السبب الجذري لأهم إصلاح)، deadband، اشكال PPO كاملة end-to-end على بيئة حقيقية.

---

## 11. الخلاصة للبدء السريع | Quick Start Summary

```bash
# 1) تدرّب
./trainctl.sh start 12

# 2) تابع (اختياري)
./trainctl.sh watch

# 3) توقف آمناً
./trainctl.sh stop

# 4) قيّم على البيانات التي لم يرها
.venv/bin/python -m src.evaluation.evaluate --checkpoint models/best.pt --split test
.venv/bin/python -m src.evaluation.evaluate --checkpoint models/best.pt --split unseen_test

# 5) استخدم النموذج (راجع القسم 5)
```

---

*آخر تحديث للوثيقة: بعد اكتمال المرحلة الأولى والثانية من التطوير — التدريب جاهز للإطلاق.*
