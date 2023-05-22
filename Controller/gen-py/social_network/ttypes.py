#
# Autogenerated by Thrift Compiler (0.16.0)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

import sys

from thrift.TRecursive import fix_spec
from thrift.Thrift import TType, TException
from thrift.transport import TTransport

all_structs = []


class ErrorCode(object):
    SE_CONNPOOL_TIMEOUT = 0
    SE_THRIFT_CONN_ERROR = 1
    SE_UNAUTHORIZED = 2
    SE_MEMCACHED_ERROR = 3
    SE_MONGODB_ERROR = 4
    SE_REDIS_ERROR = 5
    SE_THRIFT_HANDLER_ERROR = 6
    SE_RABBITMQ_CONN_ERROR = 7

    _VALUES_TO_NAMES = {
        0: "SE_CONNPOOL_TIMEOUT",
        1: "SE_THRIFT_CONN_ERROR",
        2: "SE_UNAUTHORIZED",
        3: "SE_MEMCACHED_ERROR",
        4: "SE_MONGODB_ERROR",
        5: "SE_REDIS_ERROR",
        6: "SE_THRIFT_HANDLER_ERROR",
        7: "SE_RABBITMQ_CONN_ERROR",
    }

    _NAMES_TO_VALUES = {
        "SE_CONNPOOL_TIMEOUT": 0,
        "SE_THRIFT_CONN_ERROR": 1,
        "SE_UNAUTHORIZED": 2,
        "SE_MEMCACHED_ERROR": 3,
        "SE_MONGODB_ERROR": 4,
        "SE_REDIS_ERROR": 5,
        "SE_THRIFT_HANDLER_ERROR": 6,
        "SE_RABBITMQ_CONN_ERROR": 7,
    }


class PostType(object):
    POST = 0
    REPOST = 1
    REPLY = 2
    DM = 3

    _VALUES_TO_NAMES = {
        0: "POST",
        1: "REPOST",
        2: "REPLY",
        3: "DM",
    }

    _NAMES_TO_VALUES = {
        "POST": 0,
        "REPOST": 1,
        "REPLY": 2,
        "DM": 3,
    }


class ServiceException(TException):
    """
    Attributes:
     - errorCode
     - message

    """


    def __init__(self, errorCode=None, message=None,):
        super(ServiceException, self).__setattr__('errorCode', errorCode)
        super(ServiceException, self).__setattr__('message', message)

    def __setattr__(self, *args):
        raise TypeError("can't modify immutable instance")

    def __delattr__(self, *args):
        raise TypeError("can't modify immutable instance")

    def __hash__(self):
        return hash(self.__class__) ^ hash((self.errorCode, self.message, ))

    @classmethod
    def read(cls, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and cls.thrift_spec is not None:
            return iprot._fast_decode(None, iprot, [cls, cls.thrift_spec])
        iprot.readStructBegin()
        errorCode = None
        message = None
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I32:
                    errorCode = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    message = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()
        return cls(
            errorCode=errorCode,
            message=message,
        )

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('ServiceException')
        if self.errorCode is not None:
            oprot.writeFieldBegin('errorCode', TType.I32, 1)
            oprot.writeI32(self.errorCode)
            oprot.writeFieldEnd()
        if self.message is not None:
            oprot.writeFieldBegin('message', TType.STRING, 2)
            oprot.writeString(self.message.encode('utf-8') if sys.version_info[0] == 2 else self.message)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __str__(self):
        return repr(self)

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class Media(object):
    """
    Attributes:
     - media_id
     - media_type

    """


    def __init__(self, media_id=None, media_type=None,):
        self.media_id = media_id
        self.media_type = media_type

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I64:
                    self.media_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    self.media_type = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('Media')
        if self.media_id is not None:
            oprot.writeFieldBegin('media_id', TType.I64, 1)
            oprot.writeI64(self.media_id)
            oprot.writeFieldEnd()
        if self.media_type is not None:
            oprot.writeFieldBegin('media_type', TType.STRING, 2)
            oprot.writeString(self.media_type.encode('utf-8') if sys.version_info[0] == 2 else self.media_type)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class Url(object):
    """
    Attributes:
     - shortened_url
     - expanded_url

    """


    def __init__(self, shortened_url=None, expanded_url=None,):
        self.shortened_url = shortened_url
        self.expanded_url = expanded_url

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.shortened_url = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    self.expanded_url = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('Url')
        if self.shortened_url is not None:
            oprot.writeFieldBegin('shortened_url', TType.STRING, 1)
            oprot.writeString(self.shortened_url.encode('utf-8') if sys.version_info[0] == 2 else self.shortened_url)
            oprot.writeFieldEnd()
        if self.expanded_url is not None:
            oprot.writeFieldBegin('expanded_url', TType.STRING, 2)
            oprot.writeString(self.expanded_url.encode('utf-8') if sys.version_info[0] == 2 else self.expanded_url)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class UserMention(object):
    """
    Attributes:
     - user_id
     - username

    """


    def __init__(self, user_id=None, username=None,):
        self.user_id = user_id
        self.username = username

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I64:
                    self.user_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    self.username = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('UserMention')
        if self.user_id is not None:
            oprot.writeFieldBegin('user_id', TType.I64, 1)
            oprot.writeI64(self.user_id)
            oprot.writeFieldEnd()
        if self.username is not None:
            oprot.writeFieldBegin('username', TType.STRING, 2)
            oprot.writeString(self.username.encode('utf-8') if sys.version_info[0] == 2 else self.username)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class Creator(object):
    """
    Attributes:
     - user_id
     - username

    """


    def __init__(self, user_id=None, username=None,):
        self.user_id = user_id
        self.username = username

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I64:
                    self.user_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    self.username = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('Creator')
        if self.user_id is not None:
            oprot.writeFieldBegin('user_id', TType.I64, 1)
            oprot.writeI64(self.user_id)
            oprot.writeFieldEnd()
        if self.username is not None:
            oprot.writeFieldBegin('username', TType.STRING, 2)
            oprot.writeString(self.username.encode('utf-8') if sys.version_info[0] == 2 else self.username)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class TextServiceReturn(object):
    """
    Attributes:
     - text
     - user_mentions
     - urls

    """


    def __init__(self, text=None, user_mentions=None, urls=None,):
        self.text = text
        self.user_mentions = user_mentions
        self.urls = urls

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.text = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.LIST:
                    self.user_mentions = []
                    (_etype3, _size0) = iprot.readListBegin()
                    for _i4 in range(_size0):
                        _elem5 = UserMention()
                        _elem5.read(iprot)
                        self.user_mentions.append(_elem5)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.LIST:
                    self.urls = []
                    (_etype9, _size6) = iprot.readListBegin()
                    for _i10 in range(_size6):
                        _elem11 = Url()
                        _elem11.read(iprot)
                        self.urls.append(_elem11)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('TextServiceReturn')
        if self.text is not None:
            oprot.writeFieldBegin('text', TType.STRING, 1)
            oprot.writeString(self.text.encode('utf-8') if sys.version_info[0] == 2 else self.text)
            oprot.writeFieldEnd()
        if self.user_mentions is not None:
            oprot.writeFieldBegin('user_mentions', TType.LIST, 2)
            oprot.writeListBegin(TType.STRUCT, len(self.user_mentions))
            for iter12 in self.user_mentions:
                iter12.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.urls is not None:
            oprot.writeFieldBegin('urls', TType.LIST, 3)
            oprot.writeListBegin(TType.STRUCT, len(self.urls))
            for iter13 in self.urls:
                iter13.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class Post(object):
    """
    Attributes:
     - post_id
     - creator
     - req_id
     - text
     - user_mentions
     - media
     - urls
     - timestamp
     - post_type

    """


    def __init__(self, post_id=None, creator=None, req_id=None, text=None, user_mentions=None, media=None, urls=None, timestamp=None, post_type=None,):
        self.post_id = post_id
        self.creator = creator
        self.req_id = req_id
        self.text = text
        self.user_mentions = user_mentions
        self.media = media
        self.urls = urls
        self.timestamp = timestamp
        self.post_type = post_type

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I64:
                    self.post_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRUCT:
                    self.creator = Creator()
                    self.creator.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.I64:
                    self.req_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.STRING:
                    self.text = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.LIST:
                    self.user_mentions = []
                    (_etype17, _size14) = iprot.readListBegin()
                    for _i18 in range(_size14):
                        _elem19 = UserMention()
                        _elem19.read(iprot)
                        self.user_mentions.append(_elem19)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 6:
                if ftype == TType.LIST:
                    self.media = []
                    (_etype23, _size20) = iprot.readListBegin()
                    for _i24 in range(_size20):
                        _elem25 = Media()
                        _elem25.read(iprot)
                        self.media.append(_elem25)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 7:
                if ftype == TType.LIST:
                    self.urls = []
                    (_etype29, _size26) = iprot.readListBegin()
                    for _i30 in range(_size26):
                        _elem31 = Url()
                        _elem31.read(iprot)
                        self.urls.append(_elem31)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 8:
                if ftype == TType.I64:
                    self.timestamp = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 9:
                if ftype == TType.I32:
                    self.post_type = iprot.readI32()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('Post')
        if self.post_id is not None:
            oprot.writeFieldBegin('post_id', TType.I64, 1)
            oprot.writeI64(self.post_id)
            oprot.writeFieldEnd()
        if self.creator is not None:
            oprot.writeFieldBegin('creator', TType.STRUCT, 2)
            self.creator.write(oprot)
            oprot.writeFieldEnd()
        if self.req_id is not None:
            oprot.writeFieldBegin('req_id', TType.I64, 3)
            oprot.writeI64(self.req_id)
            oprot.writeFieldEnd()
        if self.text is not None:
            oprot.writeFieldBegin('text', TType.STRING, 4)
            oprot.writeString(self.text.encode('utf-8') if sys.version_info[0] == 2 else self.text)
            oprot.writeFieldEnd()
        if self.user_mentions is not None:
            oprot.writeFieldBegin('user_mentions', TType.LIST, 5)
            oprot.writeListBegin(TType.STRUCT, len(self.user_mentions))
            for iter32 in self.user_mentions:
                iter32.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.media is not None:
            oprot.writeFieldBegin('media', TType.LIST, 6)
            oprot.writeListBegin(TType.STRUCT, len(self.media))
            for iter33 in self.media:
                iter33.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.urls is not None:
            oprot.writeFieldBegin('urls', TType.LIST, 7)
            oprot.writeListBegin(TType.STRUCT, len(self.urls))
            for iter34 in self.urls:
                iter34.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.timestamp is not None:
            oprot.writeFieldBegin('timestamp', TType.I64, 8)
            oprot.writeI64(self.timestamp)
            oprot.writeFieldEnd()
        if self.post_type is not None:
            oprot.writeFieldBegin('post_type', TType.I32, 9)
            oprot.writeI32(self.post_type)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class ComposePostOnlyFirstFourAndMiddleReturn(object):
    """
    Attributes:
     - req_id
     - user_id
     - post

    """


    def __init__(self, req_id=None, user_id=None, post=None,):
        self.req_id = req_id
        self.user_id = user_id
        self.post = post

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I64:
                    self.req_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.I64:
                    self.user_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.STRUCT:
                    self.post = Post()
                    self.post.read(iprot)
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('ComposePostOnlyFirstFourAndMiddleReturn')
        if self.req_id is not None:
            oprot.writeFieldBegin('req_id', TType.I64, 1)
            oprot.writeI64(self.req_id)
            oprot.writeFieldEnd()
        if self.user_id is not None:
            oprot.writeFieldBegin('user_id', TType.I64, 2)
            oprot.writeI64(self.user_id)
            oprot.writeFieldEnd()
        if self.post is not None:
            oprot.writeFieldBegin('post', TType.STRUCT, 3)
            self.post.write(oprot)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class ComposePostOnlyFirstFourReturn(object):
    """
    Attributes:
     - req_id
     - user_id
     - text_service_return
     - creator
     - media
     - post_id

    """


    def __init__(self, req_id=None, user_id=None, text_service_return=None, creator=None, media=None, post_id=None,):
        self.req_id = req_id
        self.user_id = user_id
        self.text_service_return = text_service_return
        self.creator = creator
        self.media = media
        self.post_id = post_id

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I64:
                    self.req_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.I64:
                    self.user_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.STRUCT:
                    self.text_service_return = TextServiceReturn()
                    self.text_service_return.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.STRUCT:
                    self.creator = Creator()
                    self.creator.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.LIST:
                    self.media = []
                    (_etype38, _size35) = iprot.readListBegin()
                    for _i39 in range(_size35):
                        _elem40 = Media()
                        _elem40.read(iprot)
                        self.media.append(_elem40)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 6:
                if ftype == TType.I64:
                    self.post_id = iprot.readI64()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('ComposePostOnlyFirstFourReturn')
        if self.req_id is not None:
            oprot.writeFieldBegin('req_id', TType.I64, 1)
            oprot.writeI64(self.req_id)
            oprot.writeFieldEnd()
        if self.user_id is not None:
            oprot.writeFieldBegin('user_id', TType.I64, 2)
            oprot.writeI64(self.user_id)
            oprot.writeFieldEnd()
        if self.text_service_return is not None:
            oprot.writeFieldBegin('text_service_return', TType.STRUCT, 3)
            self.text_service_return.write(oprot)
            oprot.writeFieldEnd()
        if self.creator is not None:
            oprot.writeFieldBegin('creator', TType.STRUCT, 4)
            self.creator.write(oprot)
            oprot.writeFieldEnd()
        if self.media is not None:
            oprot.writeFieldBegin('media', TType.LIST, 5)
            oprot.writeListBegin(TType.STRUCT, len(self.media))
            for iter41 in self.media:
                iter41.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.post_id is not None:
            oprot.writeFieldBegin('post_id', TType.I64, 6)
            oprot.writeI64(self.post_id)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(ServiceException)
ServiceException.thrift_spec = (
    None,  # 0
    (1, TType.I32, 'errorCode', None, None, ),  # 1
    (2, TType.STRING, 'message', 'UTF8', None, ),  # 2
)
all_structs.append(Media)
Media.thrift_spec = (
    None,  # 0
    (1, TType.I64, 'media_id', None, None, ),  # 1
    (2, TType.STRING, 'media_type', 'UTF8', None, ),  # 2
)
all_structs.append(Url)
Url.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'shortened_url', 'UTF8', None, ),  # 1
    (2, TType.STRING, 'expanded_url', 'UTF8', None, ),  # 2
)
all_structs.append(UserMention)
UserMention.thrift_spec = (
    None,  # 0
    (1, TType.I64, 'user_id', None, None, ),  # 1
    (2, TType.STRING, 'username', 'UTF8', None, ),  # 2
)
all_structs.append(Creator)
Creator.thrift_spec = (
    None,  # 0
    (1, TType.I64, 'user_id', None, None, ),  # 1
    (2, TType.STRING, 'username', 'UTF8', None, ),  # 2
)
all_structs.append(TextServiceReturn)
TextServiceReturn.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'text', 'UTF8', None, ),  # 1
    (2, TType.LIST, 'user_mentions', (TType.STRUCT, [UserMention, None], False), None, ),  # 2
    (3, TType.LIST, 'urls', (TType.STRUCT, [Url, None], False), None, ),  # 3
)
all_structs.append(Post)
Post.thrift_spec = (
    None,  # 0
    (1, TType.I64, 'post_id', None, None, ),  # 1
    (2, TType.STRUCT, 'creator', [Creator, None], None, ),  # 2
    (3, TType.I64, 'req_id', None, None, ),  # 3
    (4, TType.STRING, 'text', 'UTF8', None, ),  # 4
    (5, TType.LIST, 'user_mentions', (TType.STRUCT, [UserMention, None], False), None, ),  # 5
    (6, TType.LIST, 'media', (TType.STRUCT, [Media, None], False), None, ),  # 6
    (7, TType.LIST, 'urls', (TType.STRUCT, [Url, None], False), None, ),  # 7
    (8, TType.I64, 'timestamp', None, None, ),  # 8
    (9, TType.I32, 'post_type', None, None, ),  # 9
)
all_structs.append(ComposePostOnlyFirstFourAndMiddleReturn)
ComposePostOnlyFirstFourAndMiddleReturn.thrift_spec = (
    None,  # 0
    (1, TType.I64, 'req_id', None, None, ),  # 1
    (2, TType.I64, 'user_id', None, None, ),  # 2
    (3, TType.STRUCT, 'post', [Post, None], None, ),  # 3
)
all_structs.append(ComposePostOnlyFirstFourReturn)
ComposePostOnlyFirstFourReturn.thrift_spec = (
    None,  # 0
    (1, TType.I64, 'req_id', None, None, ),  # 1
    (2, TType.I64, 'user_id', None, None, ),  # 2
    (3, TType.STRUCT, 'text_service_return', [TextServiceReturn, None], None, ),  # 3
    (4, TType.STRUCT, 'creator', [Creator, None], None, ),  # 4
    (5, TType.LIST, 'media', (TType.STRUCT, [Media, None], False), None, ),  # 5
    (6, TType.I64, 'post_id', None, None, ),  # 6
)
fix_spec(all_structs)
del all_structs