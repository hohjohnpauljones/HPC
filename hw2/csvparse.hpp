using namespace std;

namespace mabcb7
{

typedef struct line {
	int x;
	int y;
	std::vector<float> points;
} Line;

std::vector<mabcb7::Line> parseFile(const std::string filename);

}

