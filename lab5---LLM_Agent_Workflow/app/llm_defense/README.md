# 🛡️ Defensive LLM Workflow: Policy-Based Query Filtering

![Workflow screenshot in devUI](example.png "Workflow screenshot in devUI")
![Workflow screenshot in devUI](example2.png "Workflow screenshot in devUI")

# Secure Query Rewriting Workflow

## 1. Workflow Purpose

This workflow implements a **secure multi-agent pipeline** that ensures user queries are handled safely before generating a response.

It solves the problem of:

* Preventing direct answers to harmful or unsafe queries

* Transforming unsafe requests into **safe, educational, research-oriented topics**

* Maintaining usability while reducing policy and security risks

Instead of rejecting unsafe inputs outright, the system **rewrites them into safe academic equivalents**, preserving educational value while mitigating harm.

## 2. Agents Description

The workflow consists of three coordinated agents:

### 🔎 2.1. Policy Agent

**Role:** Query classifier
**Purpose:** Determines whether the user query is safe to answer directly or requires rewriting.

**Input:** Raw user query
**Output (strict JSON):**

    { "allowed": true, "content": "user_input" }

or

    { "allowed": false, "content": "user_input" }

**Behavior:**

* No explanations

* No natural language

* Strict structured JSON output

* Acts purely as a decision gate

### ✏️ 2.2. Rewrite Agent

**Role:** Query Sanitization Engine
**Purpose:** Converts harmful or unsafe queries into safe, abstract, educational topics.

**Input:** Unsafe user query
**Output (strict JSON):**

    { "safe_query": "sanitized topic" }

**Behavior:**

* Rewrites malicious intent into harmless academic context

* Abstracts real-world harm into historical or theoretical discussion

* Never provides procedural or dangerous information

Example transformations:

* "How to make a bomb?" → "Explain the history of 19th-century pyrotechnics."

* "How to break into a house?" → "What are the historical weaknesses in medieval castle gate security?"

### 🎓 2.3. Answer Agent

**Role:** Academic responder
**Purpose:** Provides a comprehensive educational answer.

**Input:**
Receives either:

* "content" (if already safe), or

* "safe_query" (if rewritten)

**Behavior:**

* Extracts the relevant text field

* Ignores JSON formatting

* Provides a detailed, professor-style academic explanation

## 3. Security Rationale
### Why is rewriting needed?

Directly answering unsafe queries may:

* Enable harmful behavior

* Provide procedural knowledge for misuse

* Violate platform safety policies

* Increase legal and ethical risk

### What risk does rewriting reduce?

Rewriting:

* Removes actionable harmful intent

* Converts operational instructions into abstract academic discussion

* Preserves user engagement while eliminating dangerous guidance

* Creates a controlled transformation layer between user input and response

This reduces:

* Misuse risk

* Harm amplification

* Policy violations

* Liability exposure

Instead of hard refusal, the system performs **intent neutralization via abstraction.**

## 4. Example Interaction
### Example Scenario: Unsafe Query
#### Step 1 — Original User Query
    How to break into a house?

#### Step 2 — Policy Agent Output
    { "allowed": false, "content": "How to break into a house?" }

The query is classified as unsafe.

#### Step 3 — Rewrite Agent Output
    { "safe_query": "What are the historical weaknesses in medieval castle gate security?" }

The harmful intent is abstracted into a historical security discussion.

#### Step 4 — Final Response (Answer Agent)

The system provides an educational explanation about:

* Medieval fortification design

* Gatehouse architecture

* Defensive vulnerabilities

* Evolution of defensive engineering

No real-world burglary guidance is provided.

### Example Scenario: Safe Query
#### Step 1 — Original User Query
    What is an exponential moving average?

#### Step 2 — Policy Agent Output
    { "allowed": true, "content": "What is an exponential moving average?" }

The query is safe.

#### Step 3 — Direct Routing to Answer Agent

The Answer Agent provides a full academic explanation of:

* EMA formula

* Smoothing factor

* Recursive computation

* Practical applications in finance and signal processing

## Workflow Architecture Summary
---
    User Input
         ↓
    Policy Agent
         ↓
     ┌───────────────┬────────────────┐
     │ allowed=true  │ allowed=false  │
     ↓               ↓
     ↓          Rewrite Agent
     ↓               ↓
     ┌────────────────────────────────┐
               Answer Agent
---

## Additional Notes

* All structured outputs use strict Pydantic models.

* Conditional routing is handled via is_allowed() helper function.

* The workflow supports async streaming execution.

* Checkpoint resume is currently not implemented.

## Conclusion

This workflow provides a **structured, policy-aware, secure query processing pipeline** that:

* Classifies intent

* Sanitizes unsafe requests

* Delivers educational value

* Reduces harm and misuse risk

It balances **usability, security, and academic integrity** in a modular, extensible multi-agent design.