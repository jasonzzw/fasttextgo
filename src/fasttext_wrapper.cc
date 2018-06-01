#include <iostream>
#include <istream>
#include "fasttext.h"
#include "real.h"
#include <streambuf>
#include <cstring>

extern "C" {

struct membuf : std::streambuf
{
    membuf(char* begin, char* end) {
        this->setg(begin, begin, end);
    }
};

fasttext::FastText g_fasttext_model;
bool g_fasttext_initialized = false;

void load_model(char *path) {
  if (!g_fasttext_initialized) {
    g_fasttext_model.loadModel(std::string(path));
    g_fasttext_initialized = true;
  }
}

//get top k result
int predict(char *query, float *prob, char **buf, int *count, int k, int buf_sz) {
  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;

  g_fasttext_model.predict(in, k, predictions);

  int i=0;
  for (auto it = predictions.cbegin(); it != predictions.cend() && i<k; it++) {
    *(prob+i) = (float)exp(it->first);
    strncpy(*(buf+i), it->second.c_str(), buf_sz);
	i++;
  }
  *count=i;
  return 0;
}

}
