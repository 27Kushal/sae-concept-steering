"""Unit tests for independent quantitative judges (Python syntax, sentiment, formality)."""

import unittest
from src.evaluate import PythonSyntaxJudge, SentimentJudge, FormalityJudge


class TestJudges(unittest.TestCase):
    def test_python_syntax_judge(self):
        code_text = "def add(a, b):\n    return a + b\n\nx = add(2, 3)\nprint(x)"
        natural_text = "The river flowed gently through the ancient green forest in summer."

        score_code = PythonSyntaxJudge.score(code_text)
        score_natural = PythonSyntaxJudge.score(natural_text)

        self.assertGreater(score_code["composite_score"], score_natural["composite_score"])
        self.assertGreater(score_code["ast_validity"], 0.5)
        self.assertEqual(score_natural["ast_validity"], 0.0)

    def test_sentiment_judge(self):
        positive_text = "This is a wonderful, fantastic, and brilliant achievement. I love it!"
        negative_text = "This is a terrible, dreadful, awful, and horribly disappointing failure."

        score_pos = SentimentJudge.score(positive_text)
        score_neg = SentimentJudge.score(negative_text)

        self.assertGreater(score_pos["composite_score"], 0.6)
        self.assertLess(score_neg["composite_score"], 0.4)

    def test_formality_judge(self):
        formal_text = "Furthermore, empirical methodology demonstrates that these phenomena significantly correlate."
        informal_text = "Hey ya, so basically like it just totally worked out cool lol."

        score_formal = FormalityJudge.score(formal_text)
        score_informal = FormalityJudge.score(informal_text)

        self.assertGreater(score_formal["composite_score"], score_informal["composite_score"])


if __name__ == "__main__":
    unittest.main()
