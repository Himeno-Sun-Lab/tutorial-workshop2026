# Ratio of excitatory to inhibitory cells

There are two datasets on the ratio of excitatory to inhibitory cells that can
currently be chosen. In both datasets, the ratio of excitatory neurons given in
`fraction_E_neurons` corresponds to the ratio of excitatory cells to the sum of
excitatory and inhibitory cells.

## lichtman.csv

Data extracted from [Shapson-Coe et al. (2021) bioRxiv](https://www.biorxiv.org/content/10.1101/2021.05.29.446289v3.full.pdf), supplementary figure 5, using [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/).

Layer II (ratio 0.65) and III (ratio 0.64) are averaged for the layer II/III value.

## binzegger.csv

Values taken from from [Potjans & Diesmann 2014](https://pubmed.ncbi.nlm.nih.gov/23203991/) and calculated from these fractions:

```
fraction_E_neurons = [
    20683. / (20683. + 5834.),
    21915. / (21915. + 5479.),
    4850. / (4850. + 1065.),
    14395. / (14395. + 2948.)
]
```
