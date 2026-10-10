"""
EGX Data Sources Strategy
════════════════════════════════════════════════════════════════════════════

البورصة المصرية - مصادر البيانات الحقيقية والمستدامة
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

المشكلة الأساسية:
  - EGX لا تملك API عام مباشر
  - لكن هناك عدة مصادر رسمية وموثوقة
  - المفتاح: التنويع + التكرار (Redundancy)

الخيارات المتاحة (حسب الدقة والاستدامة):
"""

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 1: EGX Official Web Scraping (Most Reliable)
# المصدر الأساسي - موقع البورصة الرسمي
# ═══════════════════════════════════════════════════════════════════════════

"""
موقع: https://www.egx.com.eg

البيانات المتاحة:
  ✓ أسعار مباشرة (محدثة كل دقيقة خلال ساعات التداول)
  ✓ حجم التداول اليومي
  ✓ أعلى وأدنى أسعار
  ✓ الأسهم في قائمة الرابحين والخاسرين
  ✓ مؤشرات: EGX30, EGX70, EGX100
  ✓ بيانات القطاعات
  ✓ معلومات الشركات

الطريقة:
  - استخدام Selenium أو BeautifulSoup
  - جدولة الجلب (30 ثانية) خلال ساعات التداول
  - تخزين محلي (SQLite)
  - عدم الإفراط (rate limiting)

المميزات:
  + بيانات رسمية 100%
  + متحدثة في الوقت الفعلي (كل دقيقة)
  + شامل (أسهم + قطاعات + مؤشرات)
  + مستدام جدًا (الموقع الرسمي لن يختفي)

التحديات:
  - يحتاج parsing (قد يتغير تصميم الموقع)
  - يحتاج proxy في بعض الحالات
  - لا يعطي بيانات تاريخية قديمة سهلة

الحل:
  - نبني parser قوي يتعامل مع التغييرات
  - ننسخ البيانات يوميًا (backup)
  - نحتفظ بـ 5 سنوات من البيانات محليًا
"""

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 2: EGX Official Data API (Recent - 2024+)
# ═══════════════════════════════════════════════════════════════════════════

"""
المصدر: https://www.egx.com.eg/api/

البيانات:
  ✓ JSON API محدود
  ✓ بيانات سوقية حقيقية
  ✓ تحديث كل 15 دقيقة رسمي
  ✓ موثوق من البورصة الرسمية

الطريقة:
  - استدعاء HTTP GET مباشر
  - بدون authentication
  - بيانات JSON نظيفة

المميزات:
  + بيانات رسمية من EGX
  + سهل الاستخدام
  + قابل للتكرار
  + مستدام

التحديات:
  - قد لا يعطي كل الأسهم
  - قد تكون هناك حدود على الاستدعاءات
  - التوثيق محدود

الحل:
  - توثيق الـ endpoints
  - اختبار استجابة الـ API
  - backup مع Option 1
"""

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 3: Yahoo Finance (EGX Symbols)
# مصدر ثانوي موثوق
# ═══════════════════════════════════════════════════════════════════════════

"""
البيانات:
  ✓ أسعار يومية عالية الدقة
  ✓ بيانات تاريخية 20+ سنة
  ✓ حجم التداول
  ✓ معلومات الشركة

الرموز:
  - COMI.CA (Commercial International Bank)
  - TMGH.CA (Telecom Egypt)
  - FWRY.CA (Fawry)
  - ETELECOM.CA (Etisalat)
  - SWDY.CA (Swvdy for Real Estate)
  - إلخ

الطريقة:
  - yfinance library
  - محدثة يوميًا (بعد إغلاق السوق بـ 30 دقيقة)

المميزات:
  + بيانات يومية موثوقة
  + تاريخ طويل
  + مجاني
  + موثوق (الملايين يستخدمونه)

التحديات:
  - تحديث يومي فقط (ليس intraday)
  - قد تكون هناك تأخير
  - ليس للمضاربة السريعة

الحل:
  - استخدام للبيانات اليومية
  - استخدام Option 1 للـ intraday
  - دمج البيانات
"""

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 4: Investing.com Egypt
# مصدر بيانات موثوق
# ═══════════════════════════════════════════════════════════════════════════

"""
الموقع: https://eg.investing.com/

البيانات:
  ✓ أسعار مباشرة
  ✓ رسوم بيانية
  ✓ تحليلات
  ✓ أخبار EGX

الطريقة:
  - Web scraping أو استخدام Investing.com API
  - تحديث كل 5-15 دقيقة

المميزات:
  + موثوق جدًا
  + بيانات فورية
  + معلومات سياق

التحديات:
  - يحتاج parsing معقد
  - قد يكون لديهم حماية ضد السكريبنج

الحل:
  - استخدام مع rotation proxies
  - احترام robots.txt
  - backup مع مصادر أخرى
"""

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 5: ArabCrest Capital / Brokerage APIs
# مصادر متخصصة
# ═══════════════════════════════════════════════════════════════════════════

"""
هناك وسطاء مصريون يوفرون APIs:
  - ArabCrest Capital
  - EFG Hermes
  - CI Capital
  - Pharos Capital

البيانات:
  ✓ أسعار فورية
  ✓ حجم التداول الحقيقي
  ✓ أوامر معلقة

الطريقة:
  - التسجيل مع الوسيط (قد يتطلب حساب)
  - استخدام API key
  - تحديث في الوقت الفعلي

المميزات:
  + بيانات حقيقية من السوق
  + دقة عالية جدًا
  + موثوق

التحديات:
  - قد يتطلب حساب تداول
  - قد تكون هناك رسوم
  - توثيق محدود عام

الحل:
  - إذا كان لديك حساب بالفعل، استخدمه
  - للـ open source، استخدم المصادر الأخرى
"""

# ═══════════════════════════════════════════════════════════════════════════
# RECOMMENDED STRATEGY FOR EGX-MCP
# الاستراتيجية المقترحة
# ═══════════════════════════════════════════════════════════════════════════

RECOMMENDED_STRATEGY = """
PRIMARY SOURCES (الأولوية الأولى):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EGX Official Website (Live Data)
   ├─ Target: https://www.egx.com.eg
   ├─ Update: Every 1-2 minutes (during trading hours)
   ├─ Data: Prices, Volume, Indices, Sectors
   ├─ Method: Selenium + Beautiful Soup
   ├─ Storage: SQLite (real-time cache)
   └─ Priority: PRIMARY ⭐⭐⭐⭐⭐

2. Yahoo Finance (Daily + Historical)
   ├─ Target: yfinance for EGX stocks
   ├─ Update: Daily (after market close)
   ├─ Data: Daily OHLCV, 20+ year history
   ├─ Method: yfinance library
   ├─ Storage: PostgreSQL (historical archive)
   └─ Priority: SECONDARY ⭐⭐⭐⭐

SECONDARY SOURCES (النسخة الاحتياطية):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. Investing.com Egypt (Backup)
   ├─ Target: https://eg.investing.com/
   ├─ Update: Every 15 minutes
   ├─ Method: Web Scraping
   ├─ Use when: Primary source unavailable
   └─ Priority: TERTIARY ⭐⭐⭐

FALLBACK:
━━━━━━━━
4. Any Brokerage API available
   ├─ Priority: QUATERNARY ⭐⭐

DATA FLOW ARCHITECTURE:
═══════════════════════════════════════

                    ┌─────────────────────┐
                    │  EGX Official API   │
                    │ (Primary - 1-2 min) │
                    └────────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Data Aggregator      │
                    │  (Validation + Cache)   │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
         ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
         │  Real-time  │  │  Analysis   │  │  Storage   │
         │   Feed      │  │   Engine    │  │ (SQLite)   │
         └─────────────┘  └─────────────┘  └────────────┘
                │
         ┌──────▼──────────┐
         │  MCP Tools      │
         │  (Read-only)    │
         └─────────────────┘


FALLBACK CHAIN:
═══════════════

1st: EGX Official Website ✓
     ↓ (if unavailable)
2nd: Yahoo Finance Daily ✓
     ↓ (if unavailable)
3rd: Investing.com ✓
     ↓ (if all unavailable)
4th: Cached Data (historical)

DATA QUALITY CHECKS:
════════════════════

✓ Validate price ranges (no negative, no extreme jumps)
✓ Verify volume consistency
✓ Check timestamp freshness
✓ Compare across sources (3-point triangulation)
✓ Log discrepancies for manual review
✓ Alert on data quality issues

CACHING STRATEGY:
═════════════════

Real-time Cache (SQLite):
  - Last 30 days of intraday data
  - Updated every 1-2 minutes
  - Cleared daily after market close

Daily Archive (PostgreSQL):
  - Last 20 years of daily data
  - Updated once daily
  - Immutable (no overwrites)

Historical Backup:
  - Export to CSV monthly
  - Store offline
  - Disaster recovery
"""

print(RECOMMENDED_STRATEGY)

# ═══════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION PLAN
# خطة التنفيذ
# ═══════════════════════════════════════════════════════════════════════════

IMPLEMENTATION_PLAN = """
PHASE 1: Data Layer Foundation (Week 1-2)
═══════════════════════════════════════════

  1. Create data source interfaces (abstract)
  2. Implement EGX Web Scraper
     └─ Parse stock prices
     └─ Parse sector data
     └─ Parse index data
  3. Implement Yahoo Finance adapter
  4. Implement data validation
  5. Setup SQLite local cache

PHASE 2: Integration with MCP (Week 3)
═════════════════════════════════════════

  1. Create data provider service
  2. Connect to existing EGX tools
  3. Replace hardcoded data with live feed
  4. Add fallback logic

PHASE 3: Monitoring & Sustainability (Week 4+)
════════════════════════════════════════════════

  1. Add logging for data quality
  2. Setup alerts for data issues
  3. Create maintenance scripts
  4. Document data sources
  5. Setup automatic backups

REQUIRED FILES TO CREATE:
═════════════════════════

src/tradingview_mcp/core/data/
├── egx_data_source.py          # Abstract base class
├── egx_web_scraper.py          # EGX website scraper
├── egx_yahoo_adapter.py        # Yahoo Finance adapter
├── egx_investing_adapter.py    # Investing.com adapter (optional)
├── data_validator.py           # Validation & quality checks
├── data_cache.py               # SQLite caching
├── data_aggregator.py          # Primary data provider
└── data_config.py              # Configuration & endpoints

Configuration:
├── data_sources.yaml           # Source priorities
├── validation_rules.yaml       # Data quality rules
└── cache_settings.yaml         # Cache parameters
"""

print(IMPLEMENTATION_PLAN)
