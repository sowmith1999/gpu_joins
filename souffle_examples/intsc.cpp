#define SOUFFLE_GENERATOR_VERSION "2.4"
#include "souffle/CompiledSouffle.h"
#include "souffle/SignalHandler.h"
#include "souffle/SouffleInterface.h"
#include "souffle/datastructure/BTree.h"
#include "souffle/io/IOSystem.h"
#include <any>
namespace functors {
extern "C" {}
} // namespace functors
namespace souffle::t_btree_ii__0_1__11 {
using namespace souffle;
struct Type {
  static constexpr Relation::arity_type Arity = 2;
  using t_tuple = Tuple<RamDomain, 2>;
  struct t_comparator_0 {
    int operator()(const t_tuple& a, const t_tuple& b) const {
      return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1
             : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0]))
                 ? 1
                 : ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))
                        ? -1
                    : (ramBitCast<RamSigned>(a[1]) >
                       ramBitCast<RamSigned>(b[1]))
                        ? 1
                        : (0));
    }
    bool less(const t_tuple& a, const t_tuple& b) const {
      return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ||
             ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) &&
              ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
    }
    bool equal(const t_tuple& a, const t_tuple& b) const {
      return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) &&
             (ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
    }
  };
  using t_ind_0 = btree_set<t_tuple, t_comparator_0>;
  t_ind_0 ind_0;
  using iterator = t_ind_0::iterator;
  struct context {
    t_ind_0::operation_hints hints_0_lower;
    t_ind_0::operation_hints hints_0_upper;
  };
  context createContext() { return context(); }
  bool insert(const t_tuple& t);
  bool insert(const t_tuple& t, context& h);
  bool insert(const RamDomain* ramDomain);
  bool insert(RamDomain a0, RamDomain a1);
  bool contains(const t_tuple& t, context& h) const;
  bool contains(const t_tuple& t) const;
  std::size_t size() const;
  iterator find(const t_tuple& t, context& h) const;
  iterator find(const t_tuple& t) const;
  range<iterator> lowerUpperRange_00(const t_tuple& /* lower */,
                                     const t_tuple& /* upper */,
                                     context& /* h */) const;
  range<iterator> lowerUpperRange_00(const t_tuple& /* lower */,
                                     const t_tuple& /* upper */) const;
  range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower,
                                              const t_tuple& upper,
                                              context& h) const;
  range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower,
                                              const t_tuple& upper) const;
  bool empty() const;
  std::vector<range<iterator>> partition() const;
  void purge();
  iterator begin() const;
  iterator end() const;
  void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_ii__0_1__11
namespace souffle::t_btree_ii__0_1__11 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
  context h;
  return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
  if (ind_0.insert(t, h.hints_0_lower)) {
    return true;
  } else
    return false;
}
bool Type::insert(const RamDomain* ramDomain) {
  RamDomain data[2];
  std::copy(ramDomain, ramDomain + 2, data);
  const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
  context h;
  return insert(tuple, h);
}
bool Type::insert(RamDomain a0, RamDomain a1) {
  RamDomain data[2] = {a0, a1};
  return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
  return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
  context h;
  return contains(t, h);
}
std::size_t Type::size() const { return ind_0.size(); }
iterator Type::find(const t_tuple& t, context& h) const {
  return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
  context h;
  return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */,
                                         const t_tuple& /* upper */,
                                         context& /* h */) const {
  return range<iterator>(ind_0.begin(), ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */,
                                         const t_tuple& /* upper */) const {
  return range<iterator>(ind_0.begin(), ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower,
                                                  const t_tuple& upper,
                                                  context& h) const {
  t_comparator_0 comparator;
  int cmp = comparator(lower, upper);
  if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {
      fin = pos;
      ++fin;
    }
    return make_range(pos, fin);
  }
  if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
  }
  return make_range(ind_0.lower_bound(lower, h.hints_0_lower),
                    ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower,
                                                  const t_tuple& upper) const {
  context h;
  return lowerUpperRange_11(lower, upper, h);
}
bool Type::empty() const { return ind_0.empty(); }
std::vector<range<iterator>> Type::partition() const {
  return ind_0.getChunks(400);
}
void Type::purge() { ind_0.clear(); }
iterator Type::begin() const { return ind_0.begin(); }
iterator Type::end() const { return ind_0.end(); }
void Type::printStatistics(std::ostream& o) const {
  o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
  ind_0.printStats(o);
}
} // namespace souffle::t_btree_ii__0_1__11
namespace souffle {
using namespace souffle;
class Stratum_graph1_80974b3711b53e7a {
public:
  Stratum_graph1_80974b3711b53e7a(
      SymbolTable& symTable, RecordTable& recordTable,
      ConcurrentCache<std::string, std::regex>& regexCache, bool& pruneImdtRels,
      bool& performIO, SignalHandler*& signalHandler,
      std::atomic<std::size_t>& iter, std::atomic<RamDomain>& ctr,
      std::string& inputDirectory, std::string& outputDirectory,
      t_btree_ii__0_1__11::Type& rel_graph1_95de8c61a758b61d);
  void run([[maybe_unused]] const std::vector<RamDomain>& args,
           [[maybe_unused]] std::vector<RamDomain>& ret);

private:
  SymbolTable& symTable;
  RecordTable& recordTable;
  ConcurrentCache<std::string, std::regex>& regexCache;
  bool& pruneImdtRels;
  bool& performIO;
  SignalHandler*& signalHandler;
  std::atomic<std::size_t>& iter;
  std::atomic<RamDomain>& ctr;
  std::string& inputDirectory;
  std::string& outputDirectory;
  t_btree_ii__0_1__11::Type* rel_graph1_95de8c61a758b61d;
};
} // namespace  souffle
namespace souffle {
using namespace souffle;
Stratum_graph1_80974b3711b53e7a::Stratum_graph1_80974b3711b53e7a(
    SymbolTable& symTable, RecordTable& recordTable,
    ConcurrentCache<std::string, std::regex>& regexCache, bool& pruneImdtRels,
    bool& performIO, SignalHandler*& signalHandler,
    std::atomic<std::size_t>& iter, std::atomic<RamDomain>& ctr,
    std::string& inputDirectory, std::string& outputDirectory,
    t_btree_ii__0_1__11::Type& rel_graph1_95de8c61a758b61d)
    : symTable(symTable), recordTable(recordTable), regexCache(regexCache),
      pruneImdtRels(pruneImdtRels), performIO(performIO),
      signalHandler(signalHandler), iter(iter), ctr(ctr),
      inputDirectory(inputDirectory), outputDirectory(outputDirectory),
      rel_graph1_95de8c61a758b61d(&rel_graph1_95de8c61a758b61d) {}

void Stratum_graph1_80974b3711b53e7a::run(
    [[maybe_unused]] const std::vector<RamDomain>& args,
    [[maybe_unused]] std::vector<RamDomain>& ret) {
  if (performIO) {
    try {
      std::map<std::string, std::string> directiveMap(
          {{"IO", "file"},
           {"attributeNames", "x\ty"},
           {"auxArity", "0"},
           {"fact-dir", "./input"},
           {"name", "graph1"},
           {"operation", "input"},
           {"params", "{\"records\": {}, \"relation\": {\"arity\": 2, "
                      "\"params\": [\"x\", \"y\"]}}"},
           {"types",
            "{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, "
            "\"types\": [\"i:number\", \"i:number\"]}}"}});
      if (!inputDirectory.empty()) {
        directiveMap["fact-dir"] = inputDirectory;
      }
      IOSystem::getInstance()
          .getReader(directiveMap, symTable, recordTable)
          ->readAll(*rel_graph1_95de8c61a758b61d);
    } catch (std::exception& e) {
      std::cerr << "Error loading graph1 data: " << e.what() << '\n';
      exit(1);
    }
  }
}

} // namespace  souffle

namespace souffle {
using namespace souffle;
class Stratum_graph2_a48fc75a4aa69a1e {
public:
  Stratum_graph2_a48fc75a4aa69a1e(
      SymbolTable& symTable, RecordTable& recordTable,
      ConcurrentCache<std::string, std::regex>& regexCache, bool& pruneImdtRels,
      bool& performIO, SignalHandler*& signalHandler,
      std::atomic<std::size_t>& iter, std::atomic<RamDomain>& ctr,
      std::string& inputDirectory, std::string& outputDirectory,
      t_btree_ii__0_1__11::Type& rel_graph2_1b574561c23ce203);
  void run([[maybe_unused]] const std::vector<RamDomain>& args,
           [[maybe_unused]] std::vector<RamDomain>& ret);

private:
  SymbolTable& symTable;
  RecordTable& recordTable;
  ConcurrentCache<std::string, std::regex>& regexCache;
  bool& pruneImdtRels;
  bool& performIO;
  SignalHandler*& signalHandler;
  std::atomic<std::size_t>& iter;
  std::atomic<RamDomain>& ctr;
  std::string& inputDirectory;
  std::string& outputDirectory;
  t_btree_ii__0_1__11::Type* rel_graph2_1b574561c23ce203;
};
} // namespace  souffle
namespace souffle {
using namespace souffle;
Stratum_graph2_a48fc75a4aa69a1e::Stratum_graph2_a48fc75a4aa69a1e(
    SymbolTable& symTable, RecordTable& recordTable,
    ConcurrentCache<std::string, std::regex>& regexCache, bool& pruneImdtRels,
    bool& performIO, SignalHandler*& signalHandler,
    std::atomic<std::size_t>& iter, std::atomic<RamDomain>& ctr,
    std::string& inputDirectory, std::string& outputDirectory,
    t_btree_ii__0_1__11::Type& rel_graph2_1b574561c23ce203)
    : symTable(symTable), recordTable(recordTable), regexCache(regexCache),
      pruneImdtRels(pruneImdtRels), performIO(performIO),
      signalHandler(signalHandler), iter(iter), ctr(ctr),
      inputDirectory(inputDirectory), outputDirectory(outputDirectory),
      rel_graph2_1b574561c23ce203(&rel_graph2_1b574561c23ce203) {}

void Stratum_graph2_a48fc75a4aa69a1e::run(
    [[maybe_unused]] const std::vector<RamDomain>& args,
    [[maybe_unused]] std::vector<RamDomain>& ret) {
  if (performIO) {
    try {
      std::map<std::string, std::string> directiveMap(
          {{"IO", "file"},
           {"attributeNames", "x\ty"},
           {"auxArity", "0"},
           {"fact-dir", "./input"},
           {"name", "graph2"},
           {"operation", "input"},
           {"params", "{\"records\": {}, \"relation\": {\"arity\": 2, "
                      "\"params\": [\"x\", \"y\"]}}"},
           {"types",
            "{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, "
            "\"types\": [\"i:number\", \"i:number\"]}}"}});
      if (!inputDirectory.empty()) {
        directiveMap["fact-dir"] = inputDirectory;
      }
      IOSystem::getInstance()
          .getReader(directiveMap, symTable, recordTable)
          ->readAll(*rel_graph2_1b574561c23ce203);
    } catch (std::exception& e) {
      std::cerr << "Error loading graph2 data: " << e.what() << '\n';
      exit(1);
    }
  }
}

} // namespace  souffle

namespace souffle {
using namespace souffle;
class Stratum_intsc_b4d862a9a90acce3 {
public:
  Stratum_intsc_b4d862a9a90acce3(
      SymbolTable& symTable, RecordTable& recordTable,
      ConcurrentCache<std::string, std::regex>& regexCache, bool& pruneImdtRels,
      bool& performIO, SignalHandler*& signalHandler,
      std::atomic<std::size_t>& iter, std::atomic<RamDomain>& ctr,
      std::string& inputDirectory, std::string& outputDirectory,
      t_btree_ii__0_1__11::Type& rel_graph1_95de8c61a758b61d,
      t_btree_ii__0_1__11::Type& rel_graph2_1b574561c23ce203,
      t_btree_ii__0_1__11::Type& rel_intsc_475db7b637eb87bf);
  void run([[maybe_unused]] const std::vector<RamDomain>& args,
           [[maybe_unused]] std::vector<RamDomain>& ret);

private:
  SymbolTable& symTable;
  RecordTable& recordTable;
  ConcurrentCache<std::string, std::regex>& regexCache;
  bool& pruneImdtRels;
  bool& performIO;
  SignalHandler*& signalHandler;
  std::atomic<std::size_t>& iter;
  std::atomic<RamDomain>& ctr;
  std::string& inputDirectory;
  std::string& outputDirectory;
  t_btree_ii__0_1__11::Type* rel_graph1_95de8c61a758b61d;
  t_btree_ii__0_1__11::Type* rel_graph2_1b574561c23ce203;
  t_btree_ii__0_1__11::Type* rel_intsc_475db7b637eb87bf;
};
} // namespace  souffle
namespace souffle {
using namespace souffle;
Stratum_intsc_b4d862a9a90acce3::Stratum_intsc_b4d862a9a90acce3(
    SymbolTable& symTable, RecordTable& recordTable,
    ConcurrentCache<std::string, std::regex>& regexCache, bool& pruneImdtRels,
    bool& performIO, SignalHandler*& signalHandler,
    std::atomic<std::size_t>& iter, std::atomic<RamDomain>& ctr,
    std::string& inputDirectory, std::string& outputDirectory,
    t_btree_ii__0_1__11::Type& rel_graph1_95de8c61a758b61d,
    t_btree_ii__0_1__11::Type& rel_graph2_1b574561c23ce203,
    t_btree_ii__0_1__11::Type& rel_intsc_475db7b637eb87bf)
    : symTable(symTable), recordTable(recordTable), regexCache(regexCache),
      pruneImdtRels(pruneImdtRels), performIO(performIO),
      signalHandler(signalHandler), iter(iter), ctr(ctr),
      inputDirectory(inputDirectory), outputDirectory(outputDirectory),
      rel_graph1_95de8c61a758b61d(&rel_graph1_95de8c61a758b61d),
      rel_graph2_1b574561c23ce203(&rel_graph2_1b574561c23ce203),
      rel_intsc_475db7b637eb87bf(&rel_intsc_475db7b637eb87bf) {}

void Stratum_intsc_b4d862a9a90acce3::run(
    [[maybe_unused]] const std::vector<RamDomain>& args,
    [[maybe_unused]] std::vector<RamDomain>& ret) {
  signalHandler->setMsg(R"_(intsc(x,y) :- 
   graph1(x,y),
   graph2(x,y).
in file intsc.dl [10:1-10:42])_");
  if (!(rel_graph1_95de8c61a758b61d->empty()) &&
      !(rel_graph2_1b574561c23ce203->empty())) {
    [&]() {
      CREATE_OP_CONTEXT(rel_graph1_95de8c61a758b61d_op_ctxt,
                        rel_graph1_95de8c61a758b61d->createContext());
      CREATE_OP_CONTEXT(rel_graph2_1b574561c23ce203_op_ctxt,
                        rel_graph2_1b574561c23ce203->createContext());
      CREATE_OP_CONTEXT(rel_intsc_475db7b637eb87bf_op_ctxt,
                        rel_intsc_475db7b637eb87bf->createContext());
      for (const auto& env0 : *rel_graph1_95de8c61a758b61d) {
        if (rel_graph2_1b574561c23ce203->contains(
                Tuple<RamDomain, 2>{{ramBitCast(env0[0]), ramBitCast(env0[1])}},
                READ_OP_CONTEXT(rel_graph2_1b574561c23ce203_op_ctxt))) {
          Tuple<RamDomain, 2> tuple{{ramBitCast(env0[0]), ramBitCast(env0[1])}};
          rel_intsc_475db7b637eb87bf->insert(
              tuple, READ_OP_CONTEXT(rel_intsc_475db7b637eb87bf_op_ctxt));
        }
      }
    }();
  }
  if (performIO) {
    try {
      std::map<std::string, std::string> directiveMap(
          {{"IO", "file"},
           {"attributeNames", "x\ty"},
           {"auxArity", "0"},
           {"name", "intsc"},
           {"operation", "output"},
           {"output-dir", "./output"},
           {"params", "{\"records\": {}, \"relation\": {\"arity\": 2, "
                      "\"params\": [\"x\", \"y\"]}}"},
           {"types",
            "{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, "
            "\"types\": [\"i:number\", \"i:number\"]}}"}});
      if (outputDirectory == "-") {
        directiveMap["IO"] = "stdout";
        directiveMap["headers"] = "true";
      } else if (!outputDirectory.empty()) {
        directiveMap["output-dir"] = outputDirectory;
      }
      IOSystem::getInstance()
          .getWriter(directiveMap, symTable, recordTable)
          ->writeAll(*rel_intsc_475db7b637eb87bf);
    } catch (std::exception& e) {
      std::cerr << e.what();
      exit(1);
    }
  }
  if (pruneImdtRels)
    rel_graph1_95de8c61a758b61d->purge();
  if (pruneImdtRels)
    rel_graph2_1b574561c23ce203->purge();
}

} // namespace  souffle

namespace souffle {
using namespace souffle;
class Sf_intsc : public SouffleProgram {
public:
  Sf_intsc();
  ~Sf_intsc();
  void run();
  void runAll(std::string inputDirectoryArg = "",
              std::string outputDirectoryArg = "", bool performIOArg = true,
              bool pruneImdtRelsArg = true);
  void printAll([[maybe_unused]] std::string outputDirectoryArg = "");
  void loadAll([[maybe_unused]] std::string inputDirectoryArg = "");
  void dumpInputs();
  void dumpOutputs();
  SymbolTable& getSymbolTable();
  RecordTable& getRecordTable();
  void setNumThreads(std::size_t numThreadsValue);
  void executeSubroutine(std::string name, const std::vector<RamDomain>& args,
                         std::vector<RamDomain>& ret);

private:
  void runFunction(std::string inputDirectoryArg,
                   std::string outputDirectoryArg, bool performIOArg,
                   bool pruneImdtRelsArg);
  SymbolTableImpl symTable;
  SpecializedRecordTable<0> recordTable;
  ConcurrentCache<std::string, std::regex> regexCache;
  Own<t_btree_ii__0_1__11::Type> rel_graph1_95de8c61a758b61d;
  souffle::RelationWrapper<t_btree_ii__0_1__11::Type>
      wrapper_rel_graph1_95de8c61a758b61d;
  Own<t_btree_ii__0_1__11::Type> rel_graph2_1b574561c23ce203;
  souffle::RelationWrapper<t_btree_ii__0_1__11::Type>
      wrapper_rel_graph2_1b574561c23ce203;
  Own<t_btree_ii__0_1__11::Type> rel_intsc_475db7b637eb87bf;
  souffle::RelationWrapper<t_btree_ii__0_1__11::Type>
      wrapper_rel_intsc_475db7b637eb87bf;
  Stratum_graph1_80974b3711b53e7a stratum_graph1_59c3364d545da089;
  Stratum_graph2_a48fc75a4aa69a1e stratum_graph2_560ad89b77a9cfb5;
  Stratum_intsc_b4d862a9a90acce3 stratum_intsc_be0e44b58b668821;
  std::string inputDirectory;
  std::string outputDirectory;
  SignalHandler* signalHandler{SignalHandler::instance()};
  std::atomic<RamDomain> ctr{};
  std::atomic<std::size_t> iter{};
};
} // namespace  souffle
namespace souffle {
using namespace souffle;
Sf_intsc::Sf_intsc()
    : symTable(), recordTable(), regexCache(),
      rel_graph1_95de8c61a758b61d(mk<t_btree_ii__0_1__11::Type>()),
      wrapper_rel_graph1_95de8c61a758b61d(
          0, *rel_graph1_95de8c61a758b61d, *this, "graph1",
          std::array<const char*, 2>{{"i:number", "i:number"}},
          std::array<const char*, 2>{{"x", "y"}}, 0),
      rel_graph2_1b574561c23ce203(mk<t_btree_ii__0_1__11::Type>()),
      wrapper_rel_graph2_1b574561c23ce203(
          1, *rel_graph2_1b574561c23ce203, *this, "graph2",
          std::array<const char*, 2>{{"i:number", "i:number"}},
          std::array<const char*, 2>{{"x", "y"}}, 0),
      rel_intsc_475db7b637eb87bf(mk<t_btree_ii__0_1__11::Type>()),
      wrapper_rel_intsc_475db7b637eb87bf(
          2, *rel_intsc_475db7b637eb87bf, *this, "intsc",
          std::array<const char*, 2>{{"i:number", "i:number"}},
          std::array<const char*, 2>{{"x", "y"}}, 0),
      stratum_graph1_59c3364d545da089(
          symTable, recordTable, regexCache, pruneImdtRels, performIO,
          signalHandler, iter, ctr, inputDirectory, outputDirectory,
          *rel_graph1_95de8c61a758b61d),
      stratum_graph2_560ad89b77a9cfb5(
          symTable, recordTable, regexCache, pruneImdtRels, performIO,
          signalHandler, iter, ctr, inputDirectory, outputDirectory,
          *rel_graph2_1b574561c23ce203),
      stratum_intsc_be0e44b58b668821(
          symTable, recordTable, regexCache, pruneImdtRels, performIO,
          signalHandler, iter, ctr, inputDirectory, outputDirectory,
          *rel_graph1_95de8c61a758b61d, *rel_graph2_1b574561c23ce203,
          *rel_intsc_475db7b637eb87bf) {
  addRelation("graph1", wrapper_rel_graph1_95de8c61a758b61d, true, false);
  addRelation("graph2", wrapper_rel_graph2_1b574561c23ce203, true, false);
  addRelation("intsc", wrapper_rel_intsc_475db7b637eb87bf, false, true);
}

Sf_intsc::~Sf_intsc() {}

void Sf_intsc::runFunction(std::string inputDirectoryArg,
                           std::string outputDirectoryArg, bool performIOArg,
                           bool pruneImdtRelsArg) {

  this->inputDirectory = std::move(inputDirectoryArg);
  this->outputDirectory = std::move(outputDirectoryArg);
  this->performIO = performIOArg;
  this->pruneImdtRels = pruneImdtRelsArg;

  // set default threads (in embedded mode)
  // if this is not set, and omp is used, the default omp setting of number of
  // cores is used.
#if defined(_OPENMP)
  if (0 < getNumThreads()) {
    omp_set_num_threads(static_cast<int>(getNumThreads()));
  }
#endif

  signalHandler->set();
  // -- query evaluation --
  {
    std::vector<RamDomain> args, ret;
    stratum_graph1_59c3364d545da089.run(args, ret);
  }
  {
    std::vector<RamDomain> args, ret;
    stratum_graph2_560ad89b77a9cfb5.run(args, ret);
  }
  {
    std::vector<RamDomain> args, ret;
    stratum_intsc_be0e44b58b668821.run(args, ret);
  }

  // -- relation hint statistics --
  signalHandler->reset();
}

void Sf_intsc::run() { runFunction("", "", false, false); }

void Sf_intsc::runAll(std::string inputDirectoryArg,
                      std::string outputDirectoryArg, bool performIOArg,
                      bool pruneImdtRelsArg) {
  runFunction(inputDirectoryArg, outputDirectoryArg, performIOArg,
              pruneImdtRelsArg);
}

void Sf_intsc::printAll([[maybe_unused]] std::string outputDirectoryArg) {
  try {
    std::map<std::string, std::string> directiveMap(
        {{"IO", "file"},
         {"attributeNames", "x\ty"},
         {"auxArity", "0"},
         {"name", "intsc"},
         {"operation", "output"},
         {"output-dir", "./output"},
         {"params", "{\"records\": {}, \"relation\": {\"arity\": 2, "
                    "\"params\": [\"x\", \"y\"]}}"},
         {"types", "{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": "
                   "2, \"types\": [\"i:number\", \"i:number\"]}}"}});
    if (!outputDirectoryArg.empty()) {
      directiveMap["output-dir"] = outputDirectoryArg;
    }
    IOSystem::getInstance()
        .getWriter(directiveMap, symTable, recordTable)
        ->writeAll(*rel_intsc_475db7b637eb87bf);
  } catch (std::exception& e) {
    std::cerr << e.what();
    exit(1);
  }
}

void Sf_intsc::loadAll([[maybe_unused]] std::string inputDirectoryArg) {
  try {
    std::map<std::string, std::string> directiveMap(
        {{"IO", "file"},
         {"attributeNames", "x\ty"},
         {"auxArity", "0"},
         {"fact-dir", "./input"},
         {"name", "graph2"},
         {"operation", "input"},
         {"params", "{\"records\": {}, \"relation\": {\"arity\": 2, "
                    "\"params\": [\"x\", \"y\"]}}"},
         {"types", "{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": "
                   "2, \"types\": [\"i:number\", \"i:number\"]}}"}});
    if (!inputDirectoryArg.empty()) {
      directiveMap["fact-dir"] = inputDirectoryArg;
    }
    IOSystem::getInstance()
        .getReader(directiveMap, symTable, recordTable)
        ->readAll(*rel_graph2_1b574561c23ce203);
  } catch (std::exception& e) {
    std::cerr << "Error loading graph2 data: " << e.what() << '\n';
    exit(1);
  }
  try {
    std::map<std::string, std::string> directiveMap(
        {{"IO", "file"},
         {"attributeNames", "x\ty"},
         {"auxArity", "0"},
         {"fact-dir", "./input"},
         {"name", "graph1"},
         {"operation", "input"},
         {"params", "{\"records\": {}, \"relation\": {\"arity\": 2, "
                    "\"params\": [\"x\", \"y\"]}}"},
         {"types", "{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": "
                   "2, \"types\": [\"i:number\", \"i:number\"]}}"}});
    if (!inputDirectoryArg.empty()) {
      directiveMap["fact-dir"] = inputDirectoryArg;
    }
    IOSystem::getInstance()
        .getReader(directiveMap, symTable, recordTable)
        ->readAll(*rel_graph1_95de8c61a758b61d);
  } catch (std::exception& e) {
    std::cerr << "Error loading graph1 data: " << e.what() << '\n';
    exit(1);
  }
}

void Sf_intsc::dumpInputs() {
  try {
    std::map<std::string, std::string> rwOperation;
    rwOperation["IO"] = "stdout";
    rwOperation["name"] = "graph2";
    rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, "
                           "\"types\": [\"i:number\", \"i:number\"]}}";
    IOSystem::getInstance()
        .getWriter(rwOperation, symTable, recordTable)
        ->writeAll(*rel_graph2_1b574561c23ce203);
  } catch (std::exception& e) {
    std::cerr << e.what();
    exit(1);
  }
  try {
    std::map<std::string, std::string> rwOperation;
    rwOperation["IO"] = "stdout";
    rwOperation["name"] = "graph1";
    rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, "
                           "\"types\": [\"i:number\", \"i:number\"]}}";
    IOSystem::getInstance()
        .getWriter(rwOperation, symTable, recordTable)
        ->writeAll(*rel_graph1_95de8c61a758b61d);
  } catch (std::exception& e) {
    std::cerr << e.what();
    exit(1);
  }
}

void Sf_intsc::dumpOutputs() {
  try {
    std::map<std::string, std::string> rwOperation;
    rwOperation["IO"] = "stdout";
    rwOperation["name"] = "intsc";
    rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, "
                           "\"types\": [\"i:number\", \"i:number\"]}}";
    IOSystem::getInstance()
        .getWriter(rwOperation, symTable, recordTable)
        ->writeAll(*rel_intsc_475db7b637eb87bf);
  } catch (std::exception& e) {
    std::cerr << e.what();
    exit(1);
  }
}

SymbolTable& Sf_intsc::getSymbolTable() { return symTable; }

RecordTable& Sf_intsc::getRecordTable() { return recordTable; }

void Sf_intsc::setNumThreads(std::size_t numThreadsValue) {
  SouffleProgram::setNumThreads(numThreadsValue);
  symTable.setNumLanes(getNumThreads());
  recordTable.setNumLanes(getNumThreads());
  regexCache.setNumLanes(getNumThreads());
}

void Sf_intsc::executeSubroutine(std::string name,
                                 const std::vector<RamDomain>& args,
                                 std::vector<RamDomain>& ret) {
  if (name == "graph1") {
    stratum_graph1_59c3364d545da089.run(args, ret);
    return;
  }
  if (name == "graph2") {
    stratum_graph2_560ad89b77a9cfb5.run(args, ret);
    return;
  }
  if (name == "intsc") {
    stratum_intsc_be0e44b58b668821.run(args, ret);
    return;
  }
  fatal(("unknown subroutine " + name).c_str());
}

} // namespace  souffle
namespace souffle {
SouffleProgram* newInstance_intsc() { return new souffle::Sf_intsc; }
SymbolTable* getST_intsc(SouffleProgram* p) {
  return &reinterpret_cast<souffle::Sf_intsc*>(p)->getSymbolTable();
}
} // namespace souffle

#ifndef __EMBEDDED_SOUFFLE__
#include "souffle/CompiledOptions.h"
int main(int argc, char** argv) {
  try {
    souffle::CmdOptions opt(R"(intsc.dl)", R"()", R"()", false, R"()", 1);
    if (!opt.parse(argc, argv))
      return 1;
    souffle::Sf_intsc obj;
#if defined(_OPENMP)
    obj.setNumThreads(opt.getNumJobs());

#endif
    obj.runAll(opt.getInputFileDir(), opt.getOutputFileDir());
    return 0;
  } catch (std::exception& e) {
    souffle::SignalHandler::instance()->error(e.what());
  }
}
#endif

namespace souffle {
using namespace souffle;
class factory_Sf_intsc : souffle::ProgramFactory {
public:
  souffle::SouffleProgram* newInstance();
  factory_Sf_intsc();

private:
};
} // namespace  souffle
namespace souffle {
using namespace souffle;
souffle::SouffleProgram* factory_Sf_intsc::newInstance() {
  return new souffle::Sf_intsc();
}

factory_Sf_intsc::factory_Sf_intsc() : souffle::ProgramFactory("intsc") {}

} // namespace  souffle
namespace souffle {

#ifdef __EMBEDDED_SOUFFLE__
extern "C" {
souffle::factory_Sf_intsc __factory_Sf_intsc_instance;
}
#endif
} // namespace souffle
