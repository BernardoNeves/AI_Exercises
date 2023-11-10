"""
    @file exercise_Decision_Tree_class_07.py
    @author Bernardo Neves (a23494@alunos.ipca.pt)
    @brief Decision Tree implementation
    @date 2023-09-27
"""
import graphviz
from sklearn import tree
from sklearn.datasets import load_iris


class Decision_Tree:
    """A class for training and visualizing decision trees."""
    def __init__(self, dataset, data, target, criterion = "entropy", max_depth = 4, min_samples_split = 2):
        """Initialize a Decision_Tree instance."""
        
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth should be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split should be an integer greater than or equal to 2.")

        self.iris = dataset
        self.data = data
        self.target = target
        self.output_name = f"exercise_Decision_Tree_output_{criterion.capitalize()}"
        self.clf = tree.DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

    def train_decision_tree(self):
        """Train the decision tree on the provided dataset."""
        self.clf = self.clf.fit(self.data, self.target)

    def visualize_decision_tree(self):
        """Visualize the trained decision tree."""
        dot_data = tree.export_graphviz(
            self.clf,
            out_file=None,
            feature_names=self.iris.feature_names,
            class_names=self.iris.target_names,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)
        try:
            graph.render(self.output_name, format="pdf")
            graph.view()
        except Exception as error:
            print(f"An error occurred while rendering the decision tree: {error}")


if __name__ == "__main__":
    iris = load_iris()
    data, target = iris.data, iris.target

    decision_tree_entropy = Decision_Tree(iris, data, target, "entropy", 5, 2)
    decision_tree_gini = Decision_Tree(iris, data, target, "gini", 5, 2)

    decision_tree_entropy.train_decision_tree()
    decision_tree_entropy.visualize_decision_tree()

    decision_tree_gini.train_decision_tree()
    decision_tree_gini.visualize_decision_tree()
