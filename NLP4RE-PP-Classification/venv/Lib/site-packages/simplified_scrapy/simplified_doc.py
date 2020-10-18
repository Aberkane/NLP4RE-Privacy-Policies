#!/usr/bin/python
#coding=utf-8
from simplified_scrapy.core.regex_helper import *
from simplified_scrapy.core.request_helper import extractHtml, _getResponseStr
from simplified_scrapy.extracter import ExtractModel
from simplified_scrapy.core.utils import absoluteUrl
from simplified_scrapy.core.regex_dic import RegexDict
from simplified_scrapy.core.listex import List


class SimplifiedDoc(RegexDict):
    def __init__(self,
                 html=None,
                 start=None,
                 end=None,
                 before=None,
                 edit=True):
        self._editFlag = edit
        self['html'] = None
        self.last = None
        if (not html): return
        html = _getResponseStr(html)
        sec = getSection(html, start, end, before)
        if (sec): html = html[sec[0]:sec[1]]
        html = preDealHtml(html)
        self['html'] = html

    def loadHtml(self, html, start=None, end=None, before=None):
        if (not html): return
        html = _getResponseStr(html)
        sec = getSection(html, start, end, before)
        if (sec): html = html[sec[0]:sec[1]]
        html = preDealHtml(html)
        self['html'] = html

    def removeElement(self,
                      tag,
                      attr='class',
                      value=None,
                      html=None,
                      start=None,
                      end=None,
                      before=None):
        if html: html = preDealHtml(html)
        if (not html): html = self.html
        if (not html and self.last): html = self.last.html
        self['html'] = removeElement(tag, attr, value, html, start, end,
                                     before)
        return self['html']

    def removeElements(self,
                       tag,
                       attr='class',
                       value=None,
                       html=None,
                       start=None,
                       end=None,
                       before=None):
        if html: html = preDealHtml(html)
        if (not html): html = self.html
        if (not html and self.last): html = self.last.html
        while True:
            tmp = removeElement(tag, attr, value, html, start, end, before)
            if tmp != html:
                html = tmp
            else:
                break
        self['html'] = html
        return self['html']


    def getSection(self, html=None, start=None, end=None, before=None):
        return self._getSection(html, start, end, before)[0]

    def _getSection(self, html=None, start=None, end=None, before=None):
        if html: html = preDealHtml(html)
        if (not html): html = self.html
        if (not html and self.last): html = self.last.html
        s, e = getSection(html, start, end, before)
        l = 0
        if before: l = len(before)
        elif start: l = len(start)
        el = 0
        if end: el = len(end)
        if s < 0:
            s = 0
            l = 0
        if e < 0:
            e = len(html)
            el = 0
        return (html[s + l:e], s, e + el)

    def removeHtml(self, html, separator='', tags=None):
        return removeHtml(html, separator, tags)

    def trimHtml(self, html):
        return trimHtml(html)

    def absoluteUrl(self, baseUrl, url):
        return absoluteUrl(baseUrl, url)

    def getObjByModel(self, html, url=None, models=[{"Type": 3}], title=None):
        if (not isinstance(models, dict) and not isinstance(models, list)):
            models = json.loads(models)
        if (isinstance(models, dict)):
            models = [models]
        return extractHtml(url, html, models, None, title)
