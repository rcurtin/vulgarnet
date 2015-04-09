#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffnn.hpp>

using namespace mlpack;
using namespace arma;
using namespace std;

PARAM_STRING_REQ("input_file", "Corpus of text to learn on.", "i");
PARAM_INT("history", "Length of history to cache.", "H", 3);

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const size_t history = (size_t) CLI::GetParam<int>("history");

  // Now, we need to load the input dataset, convert it to character IDs, and
  // put it in an Armadillo matrix.
  vector<string> inputCorpus;
  fstream f(inputFile);
  while (!f.eof())
  {
    char buffer[1024];
    f.getline(buffer, 1024);
    inputCorpus.push_back(buffer);
  }
  f.close();

  // Okay; figure out how many data points we have.
  size_t numCharacters = 0;
  for (size_t i = 0; i < inputCorpus.size(); ++i)
    numCharacters += inputCorpus[i].size() + 1;

  // Start building our dataset.
  arma::mat dataset(history + 1, numCharacters);
  dataset.zeros();
  size_t currentCol = 0;
  for (size_t i = 0; i < inputCorpus.size(); ++i)
  {
    const string& str = inputCorpus[i];
    for (size_t j = 0; j <= str.size(); ++j)
    {
      const char c = (j == str.size()) ? '\0' : str[j];

      // Fill character.
      dataset(0, currentCol) = double(c);
      // Fill previous characters, if we can.  Careful not to underflow k.
      for (size_t k = std::min(j, history); k > 0; --k)
      {
        dataset(k, currentCol) = double(str[j - k]);
      }
      ++currentCol;
    }
  }

  // Now, we want to build a simple neural network.  Let's not start too hard.
  // How about three layers?  The first layer has 'history + 1' neurons; the
  // second can have '2 * history'; the third can have '512' (since there are
  // 256 possible outputs); the fourth can have '256', since that'll be the
  // output layer.

  // I wonder what will happen?
}
