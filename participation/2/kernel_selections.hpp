#ifndef SCOTTGS_KERNEL_SELECTIONS
#define SCOTTGS_KERNEL_SELECTIONS

#include <map>
#include <string>

namespace scottgs {

class KernelSelections
{
public:
	KernelSelections();
	~KernelSelections();
	inline std::string getKernelSourceFile(const std::string& kernelFunction) const
	{
		std::map<std::string,std::string>::const_iterator itr = kernelToSource.find(kernelFunction);
		if (itr != kernelToSource.end())
			return std::string(itr->second);
		else
			return std::string("KERNEL_FUNCTION_NOT_FOUND");
	}
	
	inline void addKernelSourceSelection(const std::string& kernelFunction, const std::string& kernelSource)
	{
		kernelToSource[kernelFunction] = kernelSource;
	}
	
private:
	std::map<std::string,std::string> kernelToSource;

}; // END: KernelSelections class


}; // END: scottgs namespace

#endif /* END: SCOTTGS_KERNEL_SELECTIONS */

