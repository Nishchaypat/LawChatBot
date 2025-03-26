import re
import tiktoken

class LegalQueryAnalyzer:
    """
    Advanced legal query analyzer using fuzzy logic to determine query intent and optimize 
    the weighting between general-purpose and specialized legal language models.
    """
    
    def __init__(self, legal_terms=None, citation_patterns=None, use_gemini=True):
        """
        Initialize the analyzer with necessary resources and fuzzy logic parameters.
        
        Args:
            legal_terms: List of legal terminology for detection
            citation_patterns: List of regex patterns for legal citations
        """
        self.legal_terms = legal_terms or self._get_default_legal_terms()
        self.citation_patterns = citation_patterns or self._get_default_citation_patterns()
        self.use_gemini = use_gemini
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize fuzzy logic system
        self._initialize_fuzzy_system()
        
    def _get_default_legal_terms(self):
        """Return expanded list of default legal terms if none provided."""
        return [
            "plaintiff", "defendant", "jurisdiction", "statutory", "litigation", "tort",
            "appellant", "respondent", "judicial", "injunction", "statute", "precedent",
            "adjudication", "jurisprudence", "liability", "damages", "breach", "discovery",
            "complaint", "pleading", "deposition", "affidavit", "indictment", "subpoena",
            "writ", "motion", "contract", "negligence", "remedy", "verdict", "hearing",
            "acquittal", "habeas corpus", "malpractice", "pro bono", "due process", "prima facie", "de facto", "de jure", "en banc", 
            "amicus curiae","brief", "burden of proof", "case law", "common law", "cause of action", "conviction", "counsel",
            "precedential", "appeal", "foreclosure", "contingent claim","sentence","voir dire","collateral","ex parte",
            "conservator","contempt","expunge","exhibit","magistrate","continuance",
            "condemnation","caseload","chambers","consumer bankruptcy","secured debt"
        ]

        
    def _get_default_citation_patterns(self):
        """Return expanded default regex patterns for legal citations if none provided."""
        return [
            r'\d+\s+U\.S\.\s+\d+',                    # US Reports citation
            r'\d+\s+S\.Ct\.\s+\d+',                   # Supreme Court Reporter
            r'\d+\s+F\.\d[d|r]d?\.\s+\d+',            # Federal Reporter
            r'\d+\s+F\.Supp\.\d?[d]?\s+\d+',          # Federal Supplement
            r'\d+\s+[A-Za-z\.]+\s+\d+',               # Generic reporter citation
            r'[A-Za-z]+\sv\.\s[A-Za-z]+',             # Case name (v. format)
            r'[0-9]{1,4}\s[A-Za-z\.]{1,15}\s[0-9]{1,4}', # General citation format
            r'\b\d{1,3}\s+[A-Za-z]+\.[A-Za-z]+\.\d{1,4}\b', # State appellate court citations (e.g., 196 Ohio App.3d 589)
            r'\b\d{1,3}\s+[A-Za-z]+\.\s+[0-9]+\b',     # Generic state court citations (e.g., 123 Cal.App. 456)
            r'\b\d+\s+U\.S\.C\.\s§\s\d+[a-z]?\b',      # US Code citations
            r'\b\d{1,2}\sC\.F\.R\.\spart\s\d+\b',      # Code of Federal Regulations citations
            r'[A-Za-z]+\sv\.\s[A-Za-z]+,\s\d+\s[A-Za-z]+\.[A-Za-z]+\.\d{1,4}'
            ]
        
    def _initialize_fuzzy_system(self):
        """Initialize the fuzzy inference system parameters and membership functions."""
        # Define fuzzy set parameters for each feature
        self.fuzzy_sets = {
            'legal_term_density': {
                'low': {'a': 0, 'b': 0, 'c': 1, 'd': 2},         # Tighter low range
                'medium': {'a': 1, 'b': 3, 'c': 5},              # More defined medium
                'high': {'a': 4, 'b': 6, 'c': 100, 'd': 100}     # Lower threshold for high
            },
            'citation_count': {
                'none': {'a': 0, 'b': 0, 'c': 0.1},              # Stricter none definition
                'few': {'a': 0, 'b': 0.5, 'c': 1.5},             # Narrower few range
                'many': {'a': 1, 'b': 2, 'c': 100, 'd': 100}     # Lower threshold for many
            },
            'structural_complexity': {
                'simple': {'a': 0, 'b': 0, 'c': 0.2, 'd': 0.3},  # Tighter simple range
                'moderate': {'a': 0.2, 'b': 0.4, 'c': 0.6},      # More defined moderate
                'complex': {'a': 0.5, 'b': 0.7, 'c': 1.0, 'd': 1.0}  # Slightly lower threshold
            },
            'jurisdiction_score': {
                'general': {'a': 0, 'b': 0, 'c': 0.2, 'd': 0.3}, # Tighter general range
                'specific': {'a': 0.2, 'b': 0.4, 'c': 1.0, 'd': 1.0}  # Lower threshold for specific
            }
        }
        
        # Define fuzzy inference rules
        # Define fuzzy inference rules with more extreme weighting
        self.fuzzy_rules = [
            # Rules favoring strong legal specialization (high Voyager weight)
            {'conditions': {'legal_term_density': 'high', 'citation_count': 'many'}, 
            'weight': 0.98},  
            {'conditions': {'legal_term_density': 'high', 'structural_complexity': 'complex'}, 
            'weight': 0.86},
            {'conditions': {'citation_count': 'many', 'structural_complexity': 'complex'}, 
            'weight': 0.89}, 
            {'conditions': {'legal_term_density': 'medium', 'citation_count': 'few', 
                        'jurisdiction_score': 'specific'}, 
            'weight': 0.82},
            
            # Rules for moderate legal specialization
            {'conditions': {'legal_term_density': 'medium', 'structural_complexity': 'moderate'}, 
            'weight': 0.62},
            {'conditions': {'legal_term_density': 'low', 'citation_count': 'few'}, 
            'weight': 0.53}, 
            
            # Rules favoring general knowledge (high Gemini weight)
            {'conditions': {'legal_term_density': 'low', 'citation_count': 'none', 
                        'structural_complexity': 'simple'}, 
            'weight': 0.22},  # Decreased from 0.35 (lower Voyager = higher Gemini)
            {'conditions': {'legal_term_density': 'low', 'jurisdiction_score': 'general'}, 
            'weight': 0.18}   # Decreased from 0.40
        ]

        
    def analyze_query(self, query):
        """
        Analyze query characteristics using fuzzy logic to determine model weights.
        
        Args:
            query: The input query string
            
        Returns:
            Dictionary with features, fuzzy memberships, weights, and query embedding
        """
        # Extract features without using spaCy
        features = {
            'legal_term_density': self._calculate_legal_term_density(query),
            'citation_count': self._count_citations(query),
            'structural_complexity': self._assess_complexity(query),
            'query_length': len(self._tokenize(query)),
            'jurisdiction_signals': self._detect_jurisdiction(query)
        }
        
        # Add jurisdiction score from signals
        features['jurisdiction_score'] = features['jurisdiction_signals']['jurisdiction_score']
        
        # Calculate fuzzy memberships for each feature
        fuzzy_memberships = self._fuzzify_features(features)
        
        # Apply fuzzy inference to determine weights
        weights = self._fuzzy_inference(fuzzy_memberships)
        
        return {
            'features': features,
            'fuzzy_memberships': fuzzy_memberships,
            'weights': weights,
            'query_embedding': None
        }
    
    def _tokenize(self, text):
        """Tokenize text using the appropriate tokenizer."""
        if self.use_gemini:
            return self.tokenizer.encode(text)
        else:
            return self.tokenizer.tokenize(text)
    
    def _calculate_legal_term_density(self, query):
        """
        Calculate the density of legal terminology in the query using fuzzy matching.
        
        Args:
            query: The input query string
            
        Returns:
            Percentage of query that consists of legal terminology
        """
        # Normalize query
        query_lower = query.lower()
        words = query_lower.split()
        total_tokens = len(words)
        
        if total_tokens == 0:
            return 0
        
        # Count exact matches
        exact_matches = sum(1 for term in self.legal_terms if term.lower() in query_lower)
        
        # Count fuzzy matches (with reduced weight)
        fuzzy_matches = 0
        for word in words:
                
            for term in self.legal_terms:
                term_lower = term.lower()
                # Skip if already counted as exact match
                if term_lower in query_lower:
                    continue
                    
                # Calculate similarity
                similarity = self._string_similarity(word, term_lower)
                if similarity >= 0.85:
                    fuzzy_matches += similarity * 0.7  # Weight fuzzy matches less
                    break
        
        # Calculate weighted density
        legal_term_count = exact_matches + fuzzy_matches
        return (legal_term_count / total_tokens) * 100
    
    def _string_similarity(self, s1, s2):
        """
        Calculate string similarity using Levenshtein distance ratio.
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            Similarity score between 0-1
        """
        # Simple implementation - in production, use libraries like rapidfuzz
        if not s1 or not s2:
            return 0
            
        # Calculate Levenshtein distance
        len_s1, len_s2 = len(s1), len(s2)
        if len_s1 > len_s2:
            s1, s2 = s2, s1
            len_s1, len_s2 = len_s2, len_s1
            
        # Initialize distance matrix
        d = list(range(len_s1 + 1))
        for i in range(1, len_s2 + 1):
            prev_d = d.copy()
            d[0] = i
            for j in range(1, len_s1 + 1):
                cost = 0 if s2[i-1] == s1[j-1] else 1
                d[j] = min(prev_d[j] + 1,      # deletion
                          d[j-1] + 1,         # insertion
                          prev_d[j-1] + cost)  # substitution
                          
        # Convert distance to similarity ratio
        max_len = max(len_s1, len_s2)
        if max_len == 0:
            return 1.0
        return 1.0 - (d[-1] / max_len)
    
    def _count_citations(self, query):
        """
        Count legal citations in the query with fuzzy pattern recognition.
        
        Args:
            query: The input query string
            
        Returns:
            Weighted count of legal citations
        """
        # Count exact citation matches
        exact_count = sum(len(re.findall(pattern, query)) for pattern in self.citation_patterns)
        
        # Look for potential malformed citations
        potential_citations = re.findall(r'(\d+\s*[A-Za-z\.]+\s*\d+)', query)
        fuzzy_count = 0
        
        for potential in potential_citations:
            # Skip if already counted as exact match
            if any(re.search(pattern, potential) for pattern in self.citation_patterns):
                continue
                
            # Check if it resembles a citation format
            if re.search(r'\d+\s*[A-Za-z\.]+\s*\d+', potential):
                fuzzy_count += 0.5  # Partial credit
        
        return exact_count + fuzzy_count
    
    def _assess_complexity(self, query):
        """
        Assess the structural complexity of the query using linguistic features.
        
        Args:
            query: The input query string
            
        Returns:
            Complexity score from 0-1
        """
        # Legal clause markers
        clause_markers = [
            "if", "when", "whereas", "notwithstanding", "provided that", 
            "subject to", "pursuant to", "without prejudice", "hereinafter"
        ]
        
        # Count legal clauses (simple approach without spaCy)
        query_lower = query.lower()
        clause_count = sum(1 for marker in clause_markers if marker in query_lower)
        
        # Check for complex legal conditionals
        has_conditionals = any(marker in query_lower for marker in clause_markers)
        
        # Basic syntactic complexity metrics without dependency parsing
        sentences = self._split_into_sentences(query)
        num_sentences = len(sentences)
        
        # Calculate average sentence length
        total_words = sum(len(sentence.split()) for sentence in sentences)
        avg_sent_length = total_words / num_sentences if num_sentences > 0 else 0
        
        # Normalize sentence length complexity
        sent_complexity = min(1.0, avg_sent_length / 35)
        
        # Estimate complexity based on punctuation density
        punctuation_count = sum(1 for char in query if char in ",.;:()[]{}") 
        punct_density = punctuation_count / len(query) if len(query) > 0 else 0
        punct_complexity = min(1.0, punct_density * 10)
        
        # Calculate overall complexity
        complexity = min(1.0, 
                        (clause_count * 0.15) + 
                        (0.25 if has_conditionals else 0) + 
                        (sent_complexity * 0.3) +
                        (punct_complexity * 0.3))
        
        return complexity
    
    def _split_into_sentences(self, text):
        """Split text into sentences without using spaCy."""
        # Simple sentence splitting using regex
        # This is not as accurate as spaCy but works for basic cases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s]
    
    def _detect_jurisdiction(self, query):
        """
        Detect jurisdictional signals in the query using regex patterns.
        
        Args:
            query: The input query string
            
        Returns:
            Dictionary of jurisdictional features with fuzzy scores
        """
        # Initialize jurisdictions with fuzzy scores
        jurisdictions = {
            'federal': 0.0,
            'state': 0.0,
            'international': 0.0,
            'specific_court': None,
            'jurisdiction_score': 0.0  # Overall score
        }
        
        # Federal jurisdiction signals
        federal_terms = [
            "federal", "U.S.", "United States", "SCOTUS", "Supreme Court", 
            "U.S.C.", "Federal Circuit", "Fed. Cir.", "federal law"
        ]
        
        # Check for federal signals
        query_lower = query.lower()
        federal_score = 0.0
        for term in federal_terms:
            if term.lower() in query_lower:
                federal_score = 1.0
                break
                
            # Check for fuzzy matches
            words = query_lower.split()
            for word in words:
                if self._string_similarity(term.lower(), word) > 0.85:
                    federal_score = max(federal_score, 0.7)
        
        jurisdictions['federal'] = federal_score
        
        # State jurisdiction signals
        state_names = [
            "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", 
            "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", 
            "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", 
            "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", 
            "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", 
            "New Hampshire", "New Jersey", "New Mexico", "New York", 
            "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", 
            "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
            "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
            "West Virginia", "Wisconsin", "Wyoming"
        ]
        
        # Check for state signals
        state_score = 0.0
        for state in state_names:
            if state in query:
                state_score = 1.0
                break
        
        # Check for state abbreviations
        if state_score < 1.0:
            state_abbr_pattern = r'\b([A-Z]{2})\b'
            potential_abbrs = re.findall(state_abbr_pattern, query)
            state_abbrs = [
                "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
            ]
            for abbr in potential_abbrs:
                if abbr in state_abbrs:
                    state_score = 1.0
                    break
        
        jurisdictions['state'] = state_score
        
        # International jurisdiction signals
        international_terms = [
            "international", "foreign", "treaty", "convention", "protocol",
            "transnational", "global", "worldwide", "UN", "United Nations"
        ]
        
        # Check for international signals
        international_score = 0.0
        for term in international_terms:
            if term.lower() in query_lower:
                international_score = 1.0
                break
                
            # Check for fuzzy matches
            words = query_lower.split()
            for word in words:
                if self._string_similarity(term.lower(), word) > 0.85:
                    international_score = max(international_score, 0.7)
        
        jurisdictions['international'] = international_score
        
        # Court-specific signals
        court_patterns = [
            "Circuit", "District Court", "Supreme Court", "Court of Appeals", 
            "Bankruptcy Court", "Tax Court", "Court of Claims"
        ]
        
        # Check for specific court mentions
        for pattern in court_patterns:
            if pattern in query:
                jurisdictions['specific_court'] = pattern
                break
        
        # Calculate overall jurisdiction specificity score
        jurisdictions['jurisdiction_score'] = max(
            federal_score,
            state_score,
            international_score,
            1.0 if jurisdictions['specific_court'] else 0.0
        )
        
        return jurisdictions
    
    def _fuzzify_features(self, features):
        """
        Calculate fuzzy membership values for each feature.
        
        Args:
            features: Dictionary of extracted feature values
            
        Returns:
            Dictionary of fuzzy membership values
        """
        memberships = {}
        
        # Fuzzify legal term density
        memberships['legal_term_density'] = {
            'low': self._trapezoid_membership(
                features['legal_term_density'],
                **self.fuzzy_sets['legal_term_density']['low']
            ),
            'medium': self._triangle_membership(
                features['legal_term_density'],
                **self.fuzzy_sets['legal_term_density']['medium']
            ),
            'high': self._trapezoid_membership(
                features['legal_term_density'],
                **self.fuzzy_sets['legal_term_density']['high']
            )
        }
        
        # Fuzzify citation count
        memberships['citation_count'] = {
            'none': self._triangle_membership(
                features['citation_count'],
                **self.fuzzy_sets['citation_count']['none']
            ),
            'few': self._triangle_membership(
                features['citation_count'],
                **self.fuzzy_sets['citation_count']['few']
            ),
            'many': self._trapezoid_membership(
                features['citation_count'],
                **self.fuzzy_sets['citation_count']['many']
            )
        }
        
        # Fuzzify structural complexity
        memberships['structural_complexity'] = {
            'simple': self._trapezoid_membership(
                features['structural_complexity'],
                **self.fuzzy_sets['structural_complexity']['simple']
            ),
            'moderate': self._triangle_membership(
                features['structural_complexity'],
                **self.fuzzy_sets['structural_complexity']['moderate']
            ),
            'complex': self._trapezoid_membership(
                features['structural_complexity'],
                **self.fuzzy_sets['structural_complexity']['complex']
            )
        }
        
        # Fuzzify jurisdiction score
        memberships['jurisdiction_score'] = {
            'general': self._trapezoid_membership(
                features['jurisdiction_score'],
                **self.fuzzy_sets['jurisdiction_score']['general']
            ),
            'specific': self._trapezoid_membership(
                features['jurisdiction_score'],
                **self.fuzzy_sets['jurisdiction_score']['specific']
            )
        }
        
        return memberships
    
    def _trapezoid_membership(self, x, a, b, c, d):
        """
        Trapezoidal membership function.
        
        Args:
            x: Input value
            a, b, c, d: Trapezoid parameters
                0 for x <= a or x >= d
                1 for b <= x <= c
                Rising from a to b
                Falling from c to d
                
        Returns:
            Membership value between 0 and 1
        """
        if x <= a or x >= d:
            return 0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1
        else:  # c < x < d
            return (d - x) / (d - c)
    
    def _triangle_membership(self, x, a, b, c):
        """
        Triangular membership function.
        
        Args:
            x: Input value
            a, b, c: Triangle parameters
                0 for x <= a or x >= c
                1 for x = b
                Rising from a to b
                Falling from b to c
                
        Returns:
            Membership value between 0 and 1
        """
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    def _fuzzy_inference(self, fuzzy_memberships):
        """
        Apply fuzzy inference rules with amplification to determine model weights.
        """
        rule_activations = []
        
        # Apply each rule
        for rule in self.fuzzy_rules:
            # Calculate rule strength (using min as T-norm for AND operation)
            strengths = []
            for feature, category in rule['conditions'].items():
                if feature in fuzzy_memberships and category in fuzzy_memberships[feature]:
                    strengths.append(fuzzy_memberships[feature][category])
            
            # Apply AND operation across all conditions (min)
            if strengths:
                rule_strength = min(strengths)
                rule_activations.append((rule_strength, rule['weight']))
        
        # If no rules activate strongly, default to mid-range weight
        if not rule_activations or max(strength for strength, _ in rule_activations) < 0.1:
            voyager_weight = 0.6  # Default weight
        else:
            # Amplify strongest rule activations
            # Square the strengths to give more weight to stronger matches
            numerator = sum((strength ** 2) * weight for strength, weight in rule_activations)
            denominator = sum((strength ** 2) for strength, _ in rule_activations)
            
            if denominator > 0:
                voyager_weight = numerator / denominator
            else:
                voyager_weight = 0.6  # Default weight

        if voyager_weight > 0.6:
            # Amplify high weights (legal queries)
            voyager_weight = 0.6 + (voyager_weight - 0.6) * 1.5
        elif voyager_weight < 0.6:
            # Amplify low weights (standard queries)
            voyager_weight = 0.6 - (0.6 - voyager_weight) * 1.5
        
        # Ensure weights are within valid range
        voyager_weight = min(max(voyager_weight, 0.1), 0.95)
        gemini_weight = 1.0 - voyager_weight
        
        return {
            'gemini': gemini_weight,
            'voyager': voyager_weight
        }