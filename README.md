# Laithinho FPL AI V6.5.2

نسخة إصلاح لمحرك الاستراتيجيات الثلاث:
- التشكيلة الآمنة
- التشكيلة المتوازنة
- تشكيلة الـDifferentials

تم إصلاح مشكلة `NameError` الخاصة بـ `_safe_build_strategy` عبر توحيد اسم الغلاف إلى `safe_build_strategy`، مع حماية إضافية حتى لا يؤدي فشل استراتيجية واحدة إلى سقوط التطبيق كاملًا.

## التشغيل
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ملاحظة
الأداة توقع احتمالي وليست ضمانًا للنتائج.
