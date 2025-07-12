#define DEFINE_NODE_KIND(Derived, KindEnum)                   \
  NodeKind getKind() const override { return KindEnum; } \
  static bool classof(const AdjNode *n) {                     \
    return n->getKind() == KindEnum;                          \
  }
#ifndef NODE_HELPERS_H
#define NODE_HELPERS_H



using IISet = llvm::SmallBitVector;
inline IISet fullSet(unsigned II){ return IISet(II,true); }
inline IISet singleton(unsigned bit,unsigned II){
  IISet s(II,false); s.set(bit); return s;
}
inline IISet union_(const IISet&a,const IISet&b){ IISet x=a; x|=b; return x; }
inline IISet intersection_(const IISet&a,const IISet&b){ IISet x=a; x&=b; return x; }
template <auto MapPtr, typename KeyT, typename Updater>
inline void updateHandshake(AdjNode *node, const KeyT &key, Updater &&updater) {
  auto &map = node->*MapPtr;                       // validSignal  readySignal …
  auto  it  = map.try_emplace(key).first;       // default:construct if absent
  updater(it->second);                                // mutate
}
template <typename KeyT>
inline void setReady(AdjNode *node, const KeyT &key, unsigned val) {
  updateHandshake<&AdjNode::readySignal>(node, key,
      [&](unsigned &slot){ slot = val; });
}
template <typename KeyT>
inline void setValid(AdjNode *node, const KeyT &key, unsigned val) {
  updateHandshake<&AdjNode::validSignal>(node, key,
      [&](unsigned &slot){ slot = val; });
}
template <typename KeyT>
inline void setVSet(AdjNode *node, const KeyT &key, IISet set) {
  updateHandshake<&AdjNode::setV>(node, key,
      [&](IISet &slot){ slot = std::move(set); });
}

template <typename KeyT>
inline void setRSet(AdjNode *node, const KeyT &key, IISet set) {
  updateHandshake<&AdjNode::setR>(node, key,
      [&](IISet &slot){ slot = std::move(set); });
}

#endif