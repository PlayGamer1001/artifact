# -*- coding: utf-8 -*-
"""Normalize stage prompts for LLM-based label normalization."""

from __future__ import annotations


NORMALIZE_SYSTEM_PROMPT = """You are a privacy taxonomy normalization assistant.
Return exactly one label from the candidate labels provided by the user.
Do not output anything other than the label."""


USER_TEMPLATE = """Your task is to normalize the current extracted text into exactly one label from the label block.

Full sentence:
{sentence}

Current extracted text:
{extracted_text}

Label block:
{label_block}

Label explanations:
{explanation_block}

Rules:
- interpret the current extracted text in the context of the full sentence
- classify only the current extracted text
- each request contains exactly one extracted element; output one label for this element only
- when the text clearly refers to a specific kind of data, prefer the specific label over Data(Unspecified)
- do not use the explanations as output; use them only to understand the labels

Output requirements:
- output exactly one label from the label block
- output only the label text
- do not explain your answer
- do not output any extra words, punctuation, or formatting
"""


LABEL_BLOCK_DATA = """
- Data(Unspecified)
- De-Identified Data
- Publicly Available Information
- Aggregated Data
- Personal Identifier
- Commercial Information
- Internet/Electronic Network Activity
- Sensory Data
- Consumer Profile
- Coarse/Approximate Location
- Professional/Employment-Related Information
- Sensitive Data
- Biometric Data
"""


EXPLANATION_BLOCK_DATA = """
- Personal Identifier: information that directly identifies a person or is mainly used to identify, contact, or recognize a person
  Examples: name, signature, alias, address, postal address, email address, physical characteristics/description, unique personal identifier, device identifier, IMEI, serial number, sim serial number, online identifier, android ID, GUID, GSFID, router ssid, internet protocol address, telephone number, beacons, pixel tags, mobile ad identifiers, customer number, user alias, unique pseudonym, education, account name, insurance policy number, medical information, health insurance information
- Commercial Information: records of property, purchases, transactions, consumption, or other commercial behavior
  Examples: personal property records, shopping history, shopping intent/tendency
- Internet/Electronic Network Activity: information about use of websites, applications, devices, networks, or digital services
  Examples: browsing history, search history, application install, consumer interaction
- Sensory Data: recorded or observed sensory information about a person or their environment
  Examples: audio information, electronic information, visual information, thermal information, olfactory information
- Consumer Profile: inferred profile, preferences, characteristics, or behavioral tendencies derived from other data
- Coarse/Approximate Location: general area or approximate location, not a precisely pinpointed location
  Examples: area code, altitude
- Professional/Employment-Related Information: information about a person's job, work status, occupation, or employment background
  Examples: employment, employment history
- Sensitive Data: especially private, high-risk, or specially regulated personal data
  Examples: government-issued identifier, social security number, driver's license number, state identification card number, passport number, origin/ethnicity, religion, union membership, message content/mail/email/text, victim status, health, financial info/financial account number/bank account number/account log-in/payment card number/security & access codes, gender identity, sex life/sexuality, citizenship, consumer health data/gender-affirming health data/reproductive or sexual health data, known child's data, precise geolocation data, genetic data, neural data
- Biometric Data: biological, physical, or behavioral data used or intended to identify a specific person
  Examples: identifier template, minutiae template, iris, retina, fingerprint, hand picture, palm picture, vein patterns, facial mapping/geometry/templates, face imagery, voiceprint, voice recording, keystroke patterns/rhythms, gait patterns/rhythms, sleep/health/exercise data, DNA/DNA-related information
- De-Identified Data: data with identifiers removed so individuals cannot be re-identified
- Publicly Available Information: information legally made public by the individual or public sources
- Aggregated Data: data combined and summarized across multiple individuals so individual identities are not discernible
- Data(Unspecified): consumer information, but does not identify the specific type of information involved
"""


EXPLANATION_BLOCK_ENTITY = """
- First-Party: the entity that directly interacts with the consumer and controls how and why the consumer's personal information is processed
- Consumer: the individual consumer directly provides the information
- Affiliates: affiliated entities such as a parent company, subsidiaries, sister companies, joint venture partners, or other companies under common control
- Third-Party(Unspecified): an outside person or entity other than the individual consumer directly, when no more specific subtype is clearly supported
- Advertising Networks: third parties that deliver, target, measure, or optimize advertisements or marketing campaigns
- Analytics Providers: third parties that collect, measure, analyze, or report usage, traffic, performance, or user behavior
- Authentication Providers: third parties that verify identity, support login, enable account access, or authenticate users
- Content Providers: third parties that supply or host content, media, communications, or embedded features
- Email Service Providers: third parties that send, receive, deliver, or manage email communications
- Government Entities: government agencies, regulators, law enforcement, courts, or other public authorities
- Internet Service Providers: providers of internet connectivity, network access, or related transmission services
- Operating Systems/Platforms: operating system providers, app stores, platform operators, or underlying platform infrastructure
- SDK Providers: third parties that provide software development kits, embedded libraries, or technical integrations, when no more specific subtype is clearly supported
- Social Networks: social media or social networking services, including social login, sharing, or related social features
- Payment Processors: third parties that process payments, billing, transactions, or payment-related fraud prevention
- Data Brokers: third parties that collect, aggregate, license, sell, or otherwise provide data about individuals from multiple sources
"""


EXPLANATION_BLOCK_PURPOSE = """
- Services: providing, operating, maintaining, supporting, delivering, or improving the product, service, account, transaction, or customer relationship
- Security: detecting, preventing, investigating, or responding to fraud, abuse, unauthorized access, security incidents, or other safety risks
- Legal: complying with laws, regulations, legal process, court orders, contractual obligations, or enforcing legal rights and obligations
- Advertising/Marketing: promoting products or services, delivering advertisements, sending marketing communications, or measuring marketing effectiveness
- Analytics/Research: analyzing trends, usage, engagement, or performance, conducting research, or generating insights to understand and improve products or business
- Personalization/Customization: tailoring content, recommendations, settings, features, or user experience based on preferences, profile, or behavior
- Merger/Acquisition: supporting a merger, acquisition, financing, asset sale, bankruptcy, reorganization, due diligence, or other corporate transaction
"""

# Flat lists for normalization whitelist matching (entity labels)
LABEL_BLOCK_ENTITY = """
- First-Party
- Consumer
- Affiliates
- Third-Party(Unspecified)
- Advertising Networks
- Analytics Providers
- Authentication Providers
- Content Providers
- Email Service Providers
- Government Entities
- Internet Service Providers
- Operating Systems/Platforms
- SDK Providers
- Social Networks
- Payment Processors
- Data Brokers
"""

LABEL_BLOCK_PURPOSE = """
- Services
- Security
- Legal
- Advertising/Marketing
- Analytics/Research
- Personalization/Customization
- Merger/Acquisition
"""


def label_block_for_field(field: str) -> str:
    if field == "data":
        return LABEL_BLOCK_DATA
    if field in ("source", "recipient", "actor"):
        return LABEL_BLOCK_ENTITY
    if field == "purpose":
        return LABEL_BLOCK_PURPOSE
    return ""


def explanation_for_field(field: str) -> str:
    if field == "data":
        return EXPLANATION_BLOCK_DATA
    if field in ("source", "recipient", "actor"):
        return EXPLANATION_BLOCK_ENTITY
    if field == "purpose":
        return EXPLANATION_BLOCK_PURPOSE
    return ""


def build_normalize_user_prompt(
    field: str,
    sentence: str,
    extracted_text: str,
    label_block: str | None = None,
) -> str:
    block = label_block if label_block is not None else label_block_for_field(field)
    expl = explanation_for_field(field)
    return USER_TEMPLATE.format(
        sentence=sentence,
        extracted_text=extracted_text,
        label_block=block,
        explanation_block=expl,
    )


def build_normalize_user_prompt_batch(
    field: str,
    sentence: str,
    extracted_text: str,
    label_block: str | None = None,
) -> str:
    return build_normalize_user_prompt(field, sentence, extracted_text, label_block)
