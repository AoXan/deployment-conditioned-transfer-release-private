# Data sources

The paper uses public or separately licensed agricultural data with a harmonised
weather interface. The source domain is US maize and the target domain is
Australian wheat. Roseworthy and other boundary datasets are described in the
Supplement.

This release contains acquisition guidance and derived tables only. Users must
obtain each raw dataset from its official provider and accept the provider's
licence before running any data-dependent reconstruction.

Expected local layout for optional raw data is `data/raw/<provider>/`. No raw data
are required for `make smoke` or `make reproduce-frozen`.
