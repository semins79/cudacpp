#include <cudacpp/String.h>
#include <string>

namespace cudacpp
{
  inline std::string & stringRef(void * const t) { return *reinterpret_cast<std::string * >(t); }
  inline std::string * stringPtr(void * const t) { return  reinterpret_cast<std::string * >(t); }
  inline const std::string & stringRef(const void * const t) { return *reinterpret_cast<const std::string * >(t); }
  inline const std::string * stringPtr(const void * const t) { return  reinterpret_cast<const std::string * >(t); }

  String::String() : internal(new std::string())
  {
  }
  String::String(const int n) : internal(new std::string(n, 0)) { }
  String::String(const int n, const char fill) : internal(new std::string(n, fill)) { }
  String::String(const char * const str) : internal(new std::string(str)) { }
  String::String(const String & rhs) : internal(new std::string(stringRef(rhs.internal))) { }
  String::~String()
  {
    delete stringPtr(internal);
  }

  String & String::operator = (const String & rhs)
  {
    stringRef(internal) = stringRef(rhs.internal);
    return *this;
  }
  String String::operator + (const String & rhs) const
  {
    String ret(*this);
    stringRef(internal) += stringRef(rhs.internal);
    return ret;
  }
  String String::operator + (const char * const rhs) const
  {
    String ret(*this);
    stringRef(ret.internal) += rhs;
    return ret;
  }
  String & String::operator += (const String & rhs)
  {
    stringRef(internal) += stringRef(rhs.internal);
    return *this;
  }
  String & String::operator += (const char * const rhs)
  {
    stringRef(internal) += rhs;
    return *this;
  }

  int String::size() const
  {
    return stringRef(internal).size();
  }
  String String::substr(const int start, const int len) const
  {
    String ret;
    stringRef(ret.internal) = stringRef(internal).substr(start, len);
    return ret;
  }
  const char * String::c_str() const
  {
    return stringRef(internal).c_str();
  }
}
