# Pro BingX Futures Live (15m • 10x • Risk 60%)
- تداول حقيقي على BingX (USDT Perpetual) عبر CCXT
- حماية TP/SL ديناميكي (ATR 1.2 SL / 1.8 TP)
- مؤشرات + رصيد حقيقي في الـ Logs و `/metrics`
- يعمل 24/7 على Render (Gunicorn + Health ping).

## الإعداد
- متغيرات البيئة: `BINGX_API_KEY`, `BINGX_API_SECRET`, `TRADE_MODE=live`, `SYMBOL=DOGE/USDT:USDT`, `INTERVAL=15m`, `LEVERAGE=10`, `RISK_ALLOC=0.6`
- `requirements.txt` يحتوي ccxt و flask و pandas.

## نشر على Render
- اربط الريبو وفيه `render.yaml`، اعمل New Web Service.
- بعد التشغيل، زر `/` و `/metrics`، وراقب **Logs**.

## ملاحظة مهمة
هذا الكود ينفذ أوامر حقيقية عند `TRADE_MODE=live`. استخدمه على مسؤوليتك، وتأكد من مفاتيح API مفعلة للفيوتشر ومحددة بصلاحيات مناسبة.
