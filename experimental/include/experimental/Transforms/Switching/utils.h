
#ifndef UTILS_HPP
#define UTILS_HPP
#include <string>
#include <string_view>
#include <memory>
#include <map>
#include <unordered_set>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"

#include "experimental/Transforms/Switching/GraphModel.h"

inline bool contains(const std::string& s, std::string_view substring) {
    return s.find(substring) != std::string::npos;
}
inline bool contains(std::string_view  s, std::string_view sub) {
  return s.find(sub) != std::string::npos;
}
inline bool contains(const std::map<std::string, std::shared_ptr<AdjGraph>>& m,
  std::string const& key)
{ return m.count(key) != 0; }

inline bool contains(const std::unordered_set<std::string>& m,
  std::string const& key)
{ return m.count(key) != 0; }


template <typename ValueT>
inline bool contains(const llvm::StringMap<ValueT>& m, llvm::StringRef key) {
  return m.find(key) != m.end();
}


// associative container(set,map etc.): check if value inside the map
template <typename Map>
inline bool contains(const Map &m,
                     const typename Map::key_type &key) {
  return m.find(key) != m.end();
}

//sequence container(vector etc.): check if value inside the container
template <typename Container, typename T>
inline bool containsValue(const Container& c, const T& value) {
    return std::find(c.begin(), c.end(), value) != c.end();
}


#endif