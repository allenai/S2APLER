"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""


import logging
import sys
import unittest
from .interface import Instance


try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning(
        """
    This test can only be run by a TIMO test runner. No tests will run.
    You may need to add this file to your project's pytest exclusions.
    """
    )
    print(e)
    sys.exit(0)


CLUSTER_SEEDS = {"1448693640485408768": {"3406298321": "require"}}

PAPERS = {
    "1591643905": {
        "authors": [
            {
                "position": 0,
                "first": "James",
                "middle": ["K."],
                "last": "Drennen",
                "suffix": None,
                "affiliations": ["Duquesne University"],
                "email": None,
            },
            {
                "position": 1,
                "first": "Randall",
                "middle": ["J."],
                "last": "Voytilla",
                "suffix": None,
                "affiliations": ["Duquesne University"],
                "email": None,
            },
        ],
        "abstract": None,
        "references": [],
        "sourced_paper_id": 1591643905,
        "source": "MAG",
        "title": "Non-Destructive Tablet Hardness Testing: The Effect of Moisture on Tablet Hardness Prediction:",
        "year": 2002,
        "doi": "10.1255/NIRN.669",
        "fieldsofstudy": ["Materials Science"],
        "publicationdate": "2002-08-01",
        "journal_name": "8-9",
        "block": "nondestructivetablethardnesstesting",
        "corpus_paper_id": 100917401,
        "source_id": "2321873794",
    },
    "3371683470": {
        "authors": [
            {
                "position": 0,
                "first": "Gustave",
                "middle": [],
                "last": "Goldmann",
                "suffix": None,
                "affiliations": [
                    "School of Sociological and Anthropological Studies, University of Ottawa, CanadaGustaveJ.Goldmann@uottawa.ca"
                ],
                "email": None,
            }
        ],
        "abstract": None,
        "references": [],
        "sourced_paper_id": 3371683470,
        "source": "TaylorAndFrancis",
        "doi": "10.1016/j.soscij.2019.01.012",
        "title": "Handbook of Social Resource Theory: Theoretical Extensions, Empirical Insights, and Social Applications",
        "year": 2019,
        "publicationdate": "2019-03-01",
        "publisher": "Taylor & Francis",
        "journal_name": "139 - 140",
        "block": "handbookofsocialresourcetheory",
        "corpus_paper_id": 212956311,
        "source_id": "10.1016/j.soscij.2019.01.012",
        "pdf_hash": "b37b0fc8b453971ea6869acd82f610b5727da4fc",
    },
    "1669949856": {
        "authors": [
            {
                "position": 0,
                "first": "HU",
                "middle": [],
                "last": "Hai-zhi",
                "suffix": None,
                "affiliations": ["Nanchang Hangkong University"],
                "email": None,
            }
        ],
        "abstract": "To distinctly find out the relationship between the rigidity of tablet and the ultrasonic speed,laser is emitted to the tablet and then we measure the speed of reflected surface acoustic wave and analyse the experimental data,finally get the tablet rigidity-wave speed curve.Experiments show that the speed will be slower and slower with the increasing rigidity of the tablet.The study is of reference value in tablet nondestructive testing or quick testing online,etc.",
        "references": [],
        "sourced_paper_id": 1669949856,
        "source": "MAG",
        "title": "Nondestructive tablet hardness testing by a new laser ultrasonic method",
        "year": 2010,
        "fieldsofstudy": ["Materials Science"],
        "journal_name": "",
        "block": "nondestructivetablethardnesstesting",
        "corpus_paper_id": 137921294,
        "source_id": "2362151143",
    },
    "1686584669": {
        "authors": [
            {
                "position": 0,
                "first": "Kjell",
                "middle": ["Y."],
                "last": "T\u00f6rnblom",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
            {
                "position": 1,
                "first": "Ali",
                "middle": [],
                "last": "Kazemi",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
        ],
        "abstract": "Handbook of social resource theory : theoretical extensions, empirical insights, and social applications",
        "references": [],
        "sourced_paper_id": 1686584669,
        "source": "MAG",
        "title": "Handbook of Social Resource Theory: Theoretical Extensions, Empirical Insights, and Social Applications",
        "year": 2012,
        "fieldsofstudy": ["Sociology"],
        "publicationdate": "2012-11-06",
        "journal_name": "",
        "block": "handbookofsocialresourcetheory",
        "corpus_paper_id": 141955816,
        "source_id": "648732766",
    },
    "3206548605": {
        "authors": [
            {
                "position": 0,
                "first": "Gustave",
                "middle": [],
                "last": "Goldmann",
                "suffix": None,
                "affiliations": ["University of Ottawa"],
                "email": None,
            }
        ],
        "abstract": None,
        "references": [],
        "sourced_paper_id": 3206548605,
        "source": "MAG",
        "title": "Handbook of Social Resource Theory: Theoretical Extensions, Empirical Insights, and Social Applications",
        "year": 2019,
        "doi": "10.1016/J.SOSCIJ.2019.01.012",
        "fieldsofstudy": ["Sociology"],
        "publicationdate": "2019-03-01",
        "journal_name": "139-140",
        "block": "handbookofsocialresourcetheory",
        "corpus_paper_id": 212956311,
        "source_id": "2996332474",
    },
    "1448693640485408768": {
        "authors": [
            {
                "position": 0,
                "first": "Gustave",
                "middle": [],
                "last": "Goldmann",
                "suffix": None,
                "affiliations": ["School of Sociological and Anthropological Studies, University of Ottawa, Canada"],
                "email": None,
            }
        ],
        "abstract": None,
        "references": [],
        "sourced_paper_id": 1448693640485408768,
        "source": "Crossref",
        "title": "Handbook of Social Resource Theory: Theoretical Extensions, Empirical Insights, and Social Applications",
        "venue": "The Social Science Journal",
        "year": 2019,
        "doi": "10.1016/j.soscij.2019.01.012",
        "publicationdate": "2019-03-01",
        "journal_name": "The Social Science Journal",
        "block": "handbookofsocialresourcetheory",
        "corpus_paper_id": 212956311,
        "source_id": "10.1016/j.soscij.2019.01.012",
    },
    "2083827223": {
        "authors": [
            {
                "position": 0,
                "first": "John D",
                "middle": [],
                "last": "Kirsch",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
            {
                "position": 1,
                "first": "James K",
                "middle": [],
                "last": "Drennen",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
        ],
        "abstract": None,
        "references": [],
        "sourced_paper_id": 2083827223,
        "source": "Unpaywall",
        "doi": "10.1016/s0731-7085(98)00132-0",
        "title": "Nondestructive tablet hardness testing by near-infrared spectroscopy: a new and robust spectral best-fit algorithm",
        "year": 1999,
        "venue": "Journal of Pharmaceutical and Biomedical Analysis",
        "journal_name": "Journal of Pharmaceutical and Biomedical Analysis",
        "block": "nondestructivetablethardnesstesting",
        "corpus_paper_id": 46562578,
        "source_id": "10.1016/s0731-7085(98)00132-0",
    },
    "1584020597": {
        "authors": [
            {
                "position": 0,
                "first": "John",
                "middle": ["D"],
                "last": "Kirsch",
                "suffix": None,
                "affiliations": ["United States Military Academy"],
                "email": None,
            },
            {
                "position": 1,
                "first": "James",
                "middle": ["K."],
                "last": "Drennen",
                "suffix": None,
                "affiliations": ["Duquesne University"],
                "email": None,
            },
        ],
        "abstract": "A new algorithm using common statistics was proposed for nondestructive near-infrared (near-IR) spectroscopic tablet hardness testing over a range of tablet potencies. The spectral features that allow near-IR tablet hardness testing were evaluated. Cimetidine tablets of 1-20% potency and 1-7 kp hardness were used for the development and testing of a new spectral best-fit algorithm for tablet hardness prediction. Actual tablet hardness values determined via a destructive diametral crushing test were used for construction of calibration models using principal component analysis/principal component regression (PCA/PCR) or the new algorithm. Both methods allowed the prediction of tablet hardness over the range of potencies studied. The spectral best-fit method compared favorably to the multivariate PCA/PCR method, but was easier to develop. The new approach offers advantages over wavelength-based regression models because the calculation of a spectral slope averages out the influence of individual spectral absorbance bands. The ability to generalize the hardness calibration over a range of potencies confirms the robust nature of the method.",
        "references": [],
        "sourced_paper_id": 1584020597,
        "source": "MAG",
        "title": "Nondestructive tablet hardness testing by near-infrared spectroscopy: a new and robust spectral best-fit algorithm.",
        "year": 1999,
        "doi": "10.1016/S0731-7085(98)00132-0",
        "fieldsofstudy": ["Chemistry"],
        "publicationdate": "1999-03-01",
        "journal_name": "351-362",
        "block": "nondestructivetablethardnesstesting",
        "corpus_paper_id": 46562578,
        "source_id": "2007050221",
    },
    "3406298321": {
        "authors": [],
        "abstract": "In an effort to be completely transparent, I wish to eclare that I am a social demographer and a statistician y training. I have no specific knowledge of Social Resource heory. Although some may consider this to be a disadvanage in reviewing this work, it in fact presents a number of lear advantages. I have no bias or preconceived notions bout the theory. I am able to view the theory through a ifferent lens \u2013 one that considers human interactions at he macro level rather than the micro level at which this heory applies. Finally, I am able to provide an objective ssessment of the structure of the handbook. A handbook can be described as a reference book on a articular subject or methodology. In the context of social cience research, a handbook must provide an overview f the theoretical framework (or frameworks), examples n which a particular methodology is applied, discussions f the strengths and weakness of the methodology and, f appropriate, directions on how to apply the methodolgy or theory. The author or authors should ideally have dopted an interdisciplinary approach to broaden the utilty of the text. The manner in which the criteria for a andbook listed above are met by this volume is elaboated upon in the following paragraphs. I consider this to e a more effective than presenting a chapter-by-chapter eview of the handbook. Does the handbook describe the methodology or theretical framework with sufficient clarity and detail that cholars in multiple disciplines may understand it? The hort answer to this question is yes. The first part of the andbook begins with a presentation of the resource theory f social exchange by Foa and Foa. They offer structure and elevance by presenting a classification of resources, definng the types of relationships through which the resources re exchanged and discussing the social relevance of the heory. This last component of the presentation is critical, ince it places the work firmly in the realm of an applied heory. T\u00f6rnblom and Kazemi add substantially to the disussion by introducing a balanced and structured critique f the work of Foa and Foa. They propose four criteria",
        "references": [],
        "sourced_paper_id": 3406298321,
        "source": "MergedPDFExtraction",
        "title": "Handbook of Social Resource Theory: Theoretical Extensions, Empirical Insights, and Social Applications. Edited by Kjell To\u0308rnblom and Ali Kazemi. Dordrecht: Springer 2014. ISBN 978-1-4939-1352-7",
        "year": 2019,
        "block": "handbookofsocialresourcetheory",
        "corpus_paper_id": 212956311,
        "source_id": "b37b0fc8b453971ea6869acd82f610b5727da4fc",
        "pdf_hash": "b37b0fc8b453971ea6869acd82f610b5727da4fc",
    },
    "2459452638": {
        "authors": [
            {
                "position": 0,
                "first": "James",
                "middle": ["K."],
                "last": "Drennen",
                "suffix": None,
                "affiliations": [
                    "Duquesne University, Graduate School of Pharmaceutical Sciences, Pittsburgh, PA 15282, USA"
                ],
                "email": "drennen@duq.edu",
            },
            {
                "position": 1,
                "first": "Randall",
                "middle": ["J."],
                "last": "Voytilla",
                "suffix": None,
                "affiliations": [
                    "Duquesne University, Graduate School of Pharmaceutical Sciences, Pittsburgh, PA 15282, USA"
                ],
                "email": None,
            },
        ],
        "abstract": None,
        "references": [],
        "sourced_paper_id": 2459452638,
        "source": "Sage",
        "doi": "10.1255/nirn.669",
        "title": "Non-Destructive Tablet Hardness Testing: The Effect of Moisture on Tablet Hardness Prediction",
        "year": 2002,
        "publicationdate": "2002-08-01",
        "journal_name": "8 - 9",
        "block": "nondestructivetablethardnesstesting",
        "corpus_paper_id": 100917401,
        "source_id": "10.1255/nirn.669",
        "pdf_hash": "0d97b326f0a3a66ff14e11871176dbadd8f735c5",
    },
    "2468186458": {
        "authors": [
            {
                "position": 0,
                "first": "James",
                "middle": ["K."],
                "last": "Drennen",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
            {
                "position": 1,
                "first": "Randall",
                "middle": ["J."],
                "last": "Voytilla",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
        ],
        "abstract": "NIR news Vol. 13 No. 4 (2002) The first publication of data demonstrating non-destructive NIR determination of tablet hardness occurred in 1991.1 Kirsch and Drennen have discussed NIR tablet hardness prediction in a review article regarding the use of NIR in the analysis of solid dosage forms.2 The character of a tablet\u2019s NIR spectrum varies little with changes in tablet hardness. However, a significant shift in the spectral baseline of a tablet occurs with increasing hardness, due to the physical changes brought about by increasing compaction force.1\u20135 Multivariate statistical techniques are commonly employed in NIR quantitative and qualitative analysis, including tablet hardness measurements, because these approaches have been proven useful for extracting desired information from NIR spectra, which often contain up to 1200 wavelengths of observation per spectrum. More recently, NIR tablet hardness determination was accomplished using the slope of the tablet spectrum.6 That work provided a simple approach for tablet hardness determination and offered predictive performance that was comparable to principal component regression (PCR). The new approach offered advantages over wavelength-based regression models because the calculation of a spectral slope averages out the influence of individual spectral absorbance bands. The current work involves an evaluation of the effect of moisture on NIR tablet hardness prediction. Water, a strong absorber over the 1100\u20132500 nm region of the NIR spectrum, is shown to cause a significant baseline shift with increasing concentrations. The result is a positive bias in the NIR predicted tablet hardness when tablets are exposed to moisture.",
        "references": [],
        "sourced_paper_id": 2468186458,
        "source": "MergedPDFExtraction",
        "title": "Non-destructive tablet hardness testing: the effect of moisture on tablet hardness prediction",
        "year": 2009,
        "block": "nondestructivetablethardnesstesting",
        "corpus_paper_id": 100917401,
        "source_id": "0d97b326f0a3a66ff14e11871176dbadd8f735c5",
        "pdf_hash": "0d97b326f0a3a66ff14e11871176dbadd8f735c5",
    },
    "2122122327": {
        "authors": [],
        "abstract": None,
        "references": [],
        "sourced_paper_id": 2122122327,
        "source": "Unpaywall",
        "doi": "10.1007/978-1-4614-4175-5",
        "title": "Handbook of Social Resource Theory",
        "year": 2012,
        "venue": "Critical Issues in Social Justice",
        "publicationtypes": [],
        "fieldsofstudy": [],
        "journal_name": "Critical Issues in Social Justice",
        "block": "handbookofsocialresourcetheory",
        "corpus_paper_id": 141955816,
        "source_id": "10.1007/978-1-4614-4175-5",
    },
    "89302044": {
        "authors": [
            {
                "position": 0,
                "first": "J",
                "middle": ["D"],
                "last": "Kirsch",
                "suffix": None,
                "affiliations": ["Merck&Company, Incorporated, West Point, PA 19486, USA."],
                "email": None,
            },
            {
                "position": 1,
                "first": "J",
                "middle": ["K"],
                "last": "Drennen",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
        ],
        "abstract": "A new algorithm using common statistics was proposed for nondestructive near-infrared (near-IR) spectroscopic tablet hardness testing over a range of tablet potencies. The spectral features that allow near-IR tablet hardness testing were evaluated. Cimetidine tablets of 1-20% potency and 1-7 kp hardness were used for the development and testing of a new spectral best-fit algorithm for tablet hardness prediction. Actual tablet hardness values determined via a destructive diametral crushing test were used for construction of calibration models using principal component analysis/principal component regression (PCA/PCR) or the new algorithm. Both methods allowed the prediction of tablet hardness over the range of potencies studied. The spectral best-fit method compared favorably to the multivariate PCA/PCR method, but was easier to develop. The new approach offers advantages over wavelength-based regression models because the calculation of a spectral slope averages out the influence of individual spectral absorbance bands. The ability to generalize the hardness calibration over a range of potencies confirms the robust nature of the method.",
        "references": [],
        "sourced_paper_id": 89302044,
        "source": "Medline",
        "pmid": 10704101,
        "title": "Nondestructive tablet hardness testing by near-infrared spectroscopy: a new and robust spectral best-fit algorithm.",
        "year": 1999,
        "venue": "Journal of pharmaceutical and biomedical analysis",
        "publicationtypes": ["JournalArticle"],
        "fieldsofstudy": ["Medicine"],
        "journal_name": "\n          351-62\n        ",
        "block": "nondestructivetablethardnesstesting",
        "corpus_paper_id": 46562578,
        "source_id": "10704101v1",
    },
}


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        instances = [Instance(papers=PAPERS, cluster_seeds=CLUSTER_SEEDS)]
        cluster_predictions = container.predict_batch(instances)[0].prediction

        # the cluster seed requirement is fulfilled
        self.assertEqual(
            set(cluster_predictions["0"]), {"1448693640485408768", "3406298321", "3371683470", "3206548605"}
        )
        # more expected results
        self.assertEqual(set(cluster_predictions["1"]), {"1591643905", "2459452638", "2468186458"})

    def test__hgm_remains_separated(self, container):
        # Setup 2 HGM source papers with the same source id. They should merge.
        hgm_sp1 = {
            "authors": [],
            "abstract": None,
            "references": [],
            "sourced_paper_id": 1,
            "source": "HumanGeneratedMetadata",
            "title": "Non-Destructive Tablet Hardness Testing",
            "year": 2002,
            "doi": "10.1255/NIRN.669",
            "fieldsofstudy": ["Materials Science"],
            "publicationdate": "2002-08-01",
            "journal_name": "8-9",
            "block": "nondestructivetablethardnesstesting",
            "corpus_paper_id": 100917401,
            "source_id": "100917401",
        }
        hgm_sp2 = hgm_sp1.copy()
        hgm_sp2["sourced_paper_id"] = 2

        same_hgm_source_id_instance = Instance(papers={"1": hgm_sp1, "2": hgm_sp2}, cluster_seeds=None)
        same_hgm_source_id_prediction = container.predict_batch([same_hgm_source_id_instance])[0].prediction.values()
        self.assertEqual(set(list(same_hgm_source_id_prediction)[0]), {"1", "2"})

        # Setup 2 HGM source papers with different source ids. They should not merge
        hgm_diff_source_id = hgm_sp1.copy()
        hgm_diff_source_id["sourced_paper_id"] = 3
        hgm_diff_source_id["doi"] = None
        hgm_diff_source_id["source_id"] = "123"

        diff_hgm_source_id_instance = Instance(papers={"1": hgm_sp1, "3": hgm_diff_source_id}, cluster_seeds=None)
        diff_hgm_source_id_prediction = container.predict_batch([diff_hgm_source_id_instance])[0].prediction.values()
        self.assertEqual(list(diff_hgm_source_id_prediction), [["1"], ["3"]])
